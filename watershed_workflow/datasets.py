import attr
import numpy as np
import collections.abc

@attr.define
class Data:
    """Simple struct for storing time-dependent rasters"""
    profile : dict
    times : np.ndarray = attr.field(converter=np.array)
    data : np.ndarray = attr.field(converter=np.array)


@attr.define
class Dataset(collections.abc.MutableMapping):
    """Stores a collection of datasets with shared times and profile."""
    profile : dict
    times : np.ndarray
    data : dict = attr.Factory(dict)

    def __getitem__(self, key):
        return Data(self.profile, self.times, self.data[key])

    def __setitem__(self, key, val):
        if isinstance(val, tuple):
            self.__setitem__(key, Data(*val))
        elif isinstance(val, Data):
            self.data[key] = val.data
        else:
            self.data[key] = np.array(val)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    def __iter__(self):
        for k in self.data:
            yield k

    def __len__(self):
        return len(self.data)

    def can_contain(self, dset):
        return (dset.profile == self.profile) and (dset.times == self.times).all()

    
class State(collections.abc.MutableMapping):
    """This is a multi-key dictionary.

    Each key is a string variable name.  Each value is a (profile,
    times, raster) tuple.  Profiles and times may be shared across
    multiple keys, hence the need for a special dictionary.

    Note that actual data is stored as a simple list of Dataset
    collections.

    """
    def __init__(self):
        self.collections = []
    
    def __getitem__(self, key):
        for col in self.collections:
            if key in col:
                return col[key]

    def __setitem__(self, key, val):
        if isinstance(val, tuple):
            self.__setitem__(key, Data(*val))
        else:
            for col in self.collections:
                if col.can_contain(val):
                    col[key] = val
                    return
            self.collections.append(Dataset(val.profile, val.times, {key : val.data}))
        
    def __delitem__(self, key):
        for col in self.collections:
            if key in col:
                col.__delitem__(key)
                break

    def __iter__(self):
        for col in self.collections:
            for k in col:
                yield k

    def __len__(self):
        return sum(len(col) for col in self.collections)




