"""Code adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/src/earthformer/datasets/sevir/sevir_dataloader.py"""

import datetime
import os

import h5pickle as h5py
import numpy as np
import numpy.random as nprand
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from .preprocessing import data_dict_to_tensor, normalize_data_dict, resize_data_dict

# SEVIR Dataset constants
SEVIR_DATA_TYPES = ["vis", "ir069", "ir107", "vil", "lght"]
SEVIR_RAW_DTYPES = {"vis": np.int16, "ir069": np.int16, "ir107": np.int16, "vil": np.uint8, "lght": np.int16}
LIGHTING_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {
    "lght": (48, 48),
}


class SEVIRTorchDataset(TorchDataset):
    """
    DataLoader that loads SEVIR sequences, and spilts each event
    into segments according to specified sequence length.

    Event Frames:
        [-----------------------raw_seq_len----------------------]
        [-----seq_len-----]
        <--stride-->[-----seq_len-----]
                    <--stride-->[-----seq_len-----]
                                        ...
    """

    def __init__(
        self,
        seq_len_in: int,
        seq_len_out: int,
        x_img_types: list[str],
        y_img_types: list[str],
        raw_seq_len: int = 49,
        sample_mode: str = "sequent",
        stride: int = 12,
        batch_size: int = 1,
        layout: str = "NHWT",
        num_shard: int = 1,
        rank: int = 0,
        split_mode: str = "uneven",
        sevir_root_dir: str = None,
        start_date: tuple[int, int, int] = None,
        end_date: tuple[int, int, int] = None,
        datetime_filter=None,
        catalog_filter="default",
        shuffle: bool = False,
        shuffle_seed: int = 1,
        output_type=np.float32,
        preprocess: bool = True,
        normalization_method: str = "01",
        downsample: dict[str, int] = None,
        upsample: dict[str, int] = None,
        verbose: bool = False,
    ):
        r"""
        Parameters
        ----------
        seq_len_in
            Length of the input sequence.
        seq_len_out
            Length of the output sequence.
        x_img_types
            Image types in the input sequence.
        y_img_types
            Image types in the output sequence.
        raw_seq_len
            The length of the raw data sequences.
        sample_mode
            'random' or 'sequent'
        stride
            Useful when sample_mode == 'sequent'
            stride must not be smaller than out_len to prevent data leakage in testing.
        batch_size
            Number of sequences in one batch.
        layout
            str: consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
            The layout of sampled data. Raw data layout is 'NHWT'.
            valid layout: 'NHWT', 'NTHW', 'NTCHW', 'TNHW', 'TNCHW'.
        num_shard
            Split the whole dataset into num_shard parts for distributed training.
        rank
            Rank of the current process within num_shard.
        split_mode: str
            if 'ceil', all `num_shard` dataloaders have the same length = ceil(total_len / num_shard).
            Different dataloaders may have some duplicated data batches, if the total size of datasets is not divided by num_shard.
            if 'floor', all `num_shard` dataloaders have the same length = floor(total_len / num_shard).
            The last several data batches may be wasted, if the total size of datasets is not divided by num_shard.
            if 'uneven', the last datasets has larger length when the total length is not divided by num_shard.
            The uneven split leads to synchronization error in dist.all_reduce() or dist.barrier().
            See related issue: https://github.com/pytorch/pytorch/issues/33148
            Notice: this also affects the behavior of `self.use_up`.
        sevir_root_dir
            Absolute path to the root directory of the SEVIR dataset.
        start_date
            Start time of SEVIR samples to generate.
        end_date
            End time of SEVIR samples to generate.
        datetime_filter
            function
            Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
        catalog_filter
            function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle
            bool, If True, data samples are shuffled before each epoch.
        shuffle_seed
            int, Seed to use for shuffling.
        output_type
            np.dtype, dtype of generated tensors
        preprocess
            bool, If True, self.preprocess_data_dict(data_dict) is called before each sample generated
        normalization_method
            Method for normalizing the data during preprocessing. Available methods are: '01', 'sevir' and ''
            (no normalization). For more detailed description check normalize_data_dict().
        downsample
            downsample.keys() == data_types, the values are downsampling factors for both height and width
            (both dimensions are downsampled). For example factor of 2 means the height and width will be 2x smaller.
        upsample
            downsample.keys() == data_types, the values are upsampling factors for both height and width
            (both dimensions are upsampled). For example factor of 2 means the height and width will be 2x larger.
        verbose
            bool, verbose when opening raw data files

        """
        super(SEVIRTorchDataset, self).__init__()

        sevir_root_dir = os.path.abspath(sevir_root_dir)
        sevir_catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
        self.sevir_data_dir = os.path.join(sevir_root_dir, "data")

        self.x_data_types = x_img_types
        self.y_data_types = y_img_types
        data_types = set(x_img_types).union(set(y_img_types))

        if not data_types:
            data_types = SEVIR_DATA_TYPES

        assert set(data_types).issubset(SEVIR_DATA_TYPES)

        # configs which should not be modified
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        self.raw_seq_len = raw_seq_len
        assert (
            seq_len_in + seq_len_out <= self.raw_seq_len
        ), f"seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len_in + seq_len_out}."
        self.seq_len = seq_len_in + seq_len_out
        assert sample_mode in [
            "random",
            "sequent",
        ], f"Invalid sample_mode = {sample_mode}, must be 'random' or 'sequent'."
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ("NHWT", "NTHW", "NTCHW", "NTHWC", "TNHW", "TNCHW")
        if layout not in valid_layout:
            raise ValueError(f"Invalid layout = {layout}! Must be one of {valid_layout}.")
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ("ceil", "floor", "uneven")
        if split_mode not in valid_split_mode:
            raise ValueError(f"Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}.")
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        self.catalog = pd.read_csv(sevir_catalog_path, parse_dates=["time_utc"], low_memory=False)
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = datetime.datetime(*start_date)
        self.end_date = datetime.datetime(*end_date)
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample
        self.upsample_dict = upsample
        self.normalization_method = normalization_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == "default":
                self.catalog_filter = self._default_catalog_filter
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files(verbose=self.verbose)
        self.reset()

    @staticmethod
    def _default_catalog_filter(c):
        return c.pct_missing == 0

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[np.logical_or.reduce([self.catalog.img_type == i for i in imgt])]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby("id").filter(lambda x: imgts.issubset(set(x["img_type"])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        filtcat = filtcat.groupby("id").filter(lambda x: x.shape[0] == len(imgt))
        self._samples = filtcat.groupby("id").apply(lambda df: self._df_to_series(df, imgt))
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index("img_type")
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != "lght" else s.id
            d.update({f"{i}_filename": [s.file_name], f"{i}_index": [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """Opens HDF files"""
        imgt = self.data_types
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique(self._samples[f"{t}_filename"].values))
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print("Opening HDF5 file for reading", f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + "/" + f, "r")

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == "ceil":
            _last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == "floor":
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """
        imgtyps = np.unique([x.split("_")[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f"{t}_filename"]
            idx = row[f"{t}_index"]
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            if t == "lght":
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx : idx + 1, :, :, t_slice]
            data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (
            (*self.data_shape["lght"], len(self.lght_frame_times))
            if t_slice.stop is None
            else (*self.data_shape["lght"], 1)
        )
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(
                        t >= self.lght_frame_times[t_slice.stop - 1], t < self.lght_frame_times[t_slice.stop]
                    )
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(t >= self.lght_frame_times[0], t < self.lght_frame_times[1])
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]

    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    def inc_sample_count(self):
        self._sample_count += 1

    @property
    def curr_event_idx(self):
        return self._curr_event_idx

    @property
    def curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self._curr_seq_idx

    def set_curr_event_idx(self, val):
        self._curr_event_idx = val

    def set_curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self._curr_seq_idx = val

    def reset(self, shuffle: bool = None):
        self.set_curr_event_idx(val=self.start_event_idx)
        self.set_curr_seq_idx(0)
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    @property
    def use_up(self):
        """
        Check if dataset is used up in 'sequent' mode.
        """
        if self.sample_mode == "random":
            return False
        else:  # self.sample_mode == 'sequent'
            # compute the remaining number of sequences in current event
            curr_event_remain_seq = self.num_seq_per_event - self.curr_seq_idx
            all_remain_seq = (
                curr_event_remain_seq + (self.end_event_idx - self.curr_event_idx - 1) * self.num_seq_per_event
            )
            if self.split_mode == "floor":
                # This approach does not cover all available data, but avoid dealing with masks
                return all_remain_seq < self.batch_size
            else:
                return all_remain_seq <= 0

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Parameters
        ----------
        event_idx
        event_batch_size
            event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns
        -------
        event_batch
            list of event batches.
            event_batch[i] is the event batch of the i-th data type.
            Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_len)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for index, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [
                    pad_size,
                ] + list(data[t].shape[1:])
                data_pad = np.concatenate(
                    (data[t].astype(self.output_type), np.zeros(pad_shape, dtype=self.output_type)), axis=0
                )
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_mode == "random":
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            else:
                self.inc_sample_count()
                ret_dict = self._sequent_sample()

        ret_dict = data_dict_to_tensor(data_dict=ret_dict, data_types=self.data_types)
        if self.preprocess:
            ret_dict = normalize_data_dict(
                data_dict=ret_dict, img_types=self.data_types, method=self.normalization_method
            )

        if self.downsample_dict or self.upsample_dict:
            ret_dict = resize_data_dict(
                data_dict=ret_dict, downsample_dict=self.downsample_dict, upsample_dict=self.upsample_dict
            )
        return ret_dict

    def __getitem__(self, index):
        return self._idx_sample(index=index)

    def _random_sample(self):
        """
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx, high=self.end_event_idx, size=self.batch_size)
        seq_idx_list = nprand.randint(low=0, high=self.num_seq_per_event, size=self.batch_size)
        seq_slice_list = [
            slice(seq_idx * self.stride, seq_idx * self.stride + self.seq_len) for seq_idx in seq_idx_list
        ]
        ret_dict = {}
        while num_sampled < self.batch_size:
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled], event_batch_size=1)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event[imgt_idx][
                    [
                        0,
                    ],
                    :,
                    :,
                    seq_slice_list[num_sampled],
                ]  # keep the dim of batch_size for concatenation
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
        return ret_dict

    def _sequent_sample(self):
        """
        Returns
        -------
        ret_dict: dict
            `ret_dict.keys()` contains `self.data_types`.
            `ret_dict["mask"]` is a list of bool, indicating if the data entry is real or padded.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        assert not self.use_up, "Data loader used up! Reset it to reuse."
        event_idx = self.curr_event_idx
        seq_idx = self.curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)
        ret_dict = {"mask": []}
        all_no_pad_flag = True
        for sampled_idx in sampled_idx_list:
            batch_slice = [
                sampled_idx["event_idx"] - start_event_idx,
            ]  # use [] to keepdim
            seq_slice = slice(sampled_idx["seq_idx"] * self.stride, sampled_idx["seq_idx"] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
            # add mask
            no_pad_flag = sampled_idx["event_idx"] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)
        if all_no_pad_flag:
            # if there is no padded data items at all, set `ret_dict["mask"] = None` for convenience.
            ret_dict["mask"] = None
        # update current idx
        self.set_curr_event_idx(event_idx)
        self.set_curr_seq_idx(seq_idx)
        return ret_dict

    def _idx_sample(self, index):
        """
        Parameters
        ----------
        index
            The index of the batch to sample.
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx, event_batch_size=event_batch_size)
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [
                sampled_idx["event_idx"] - start_event_idx,
            ]  # use [] to keepdim
            seq_slice = slice(sampled_idx["seq_idx"] * self.stride, sampled_idx["seq_idx"] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq), axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})

        ret_dict = data_dict_to_tensor(data_dict=ret_dict, data_types=self.data_types)

        # change images to (C, H, W) - this is a quick fix for single-image sequences, where the sequence dimension
        # is squeezed and batch dimension passed off as channel
        for key in ret_dict:
            ret_dict[key] = ret_dict[key].squeeze(3)

        if self.preprocess:
            ret_dict = normalize_data_dict(
                data_dict=ret_dict, img_types=self.data_types, method=self.normalization_method
            )

        if self.downsample_dict or self.upsample_dict:
            ret_dict = resize_data_dict(
                data_dict=ret_dict, downsample_dict=self.downsample_dict, upsample_dict=self.upsample_dict
            )

        data_x = torch.cat([ret_dict[t] for t in self.x_data_types], dim=0).contiguous()
        data_y = torch.cat([ret_dict[t] for t in self.y_data_types], dim=0).contiguous()

        return data_x, data_y
