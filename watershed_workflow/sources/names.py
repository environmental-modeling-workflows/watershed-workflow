"""A utility class for generating file and folder names."""

import attr
import sys, os
import workflow.conf

@attr.s
class Names:
    """File system meta data for downloading a file."""
    name = attr.ib(type=str)
    base_folder = attr.ib(type=str)
    folder_template = attr.ib(type=str)
    file_template = attr.ib(type=str)
    raw_template = attr.ib(type=str, default=None)

    def data_dir(self):
        return os.path.join(workflow.conf.rcParams['DEFAULT']['data_directory'], self.base_folder)
    
    def folder_name(self, *args, **kwargs):
        if self.folder_template is None:
            return os.path.join(self.data_dir())
        else:
            return os.path.join(self.data_dir(), self.folder_template.format(*args, **kwargs))

    def raw_folder_name(self, *args, **kwargs):
        if self.raw_template is None:
            if self.folder_template is None:
                self.raw_template = 'raw'
            else:
                self.raw_template = os.path.join(self.folder_template, 'raw')
        return os.path.join(self.data_dir(), self.raw_template.format(*args, **kwargs))
    
    def file_name_base(self, *args, **kwargs):
        return self.file_template.format(*args, **kwargs)

    def file_name(self, *args, **kwargs):
        return os.path.join(self.folder_name(*args, **kwargs), self.file_name_base(*args, **kwargs))
