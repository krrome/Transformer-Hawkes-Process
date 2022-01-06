import numpy as np
import torch
import torch.utils.data
import h5py

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file

    def __getitem__(self, index):
        fh = self.file
        return fh["timestamps"][index, :][:], fh["types"][index, :][:]

    def __len__(self):
        return len(self.file["timestamps"])


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, event_type


def collate_fn_h5(arrays):
    time, event_type = list(zip(*arrays))
    time = torch.tensor(np.array(time), dtype=torch.float32)
    event_type = torch.tensor(np.array(event_type), dtype=torch.long)
    sel = (time != 0).all(axis=0)
    return time[:, sel], event_type[:, sel]


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl


def get_dataloader_h5(h5_path, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = H5Dataset(h5_path)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn_h5,
        shuffle=shuffle
    )
    return dl
