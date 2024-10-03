import os
from typing import TextIO, Type, Union

import fsspec
import s3fs
from pydantic import BaseModel

PATH_TYPE = Union[str, bytes, os.PathLike]


class FileIOBase(BaseModel):
    @classmethod
    def from_string(cls, text: str):
        return cls()

    @classmethod
    def from_stream(cls, stream: TextIO):
        return cls.from_string(stream.read())

    def to_stream(self, stream: TextIO) -> None:
        stream.write(str(self))

    @classmethod
    def from_file(cls, file_path: PATH_TYPE):
        with open(file_path, "r") as file:
            return cls.from_stream(file)

    def to_file(self, file_path: PATH_TYPE) -> None:
        with open(file_path, "w") as file:
            self.to_stream(file)

    @classmethod
    def from_s3(cls, s3_path: str, **kwargs):
        fs = s3fs.S3FileSystem(**kwargs)
        with fs.open(s3_path, "r") as file:
            return cls.from_stream(file)

    def to_s3(self, s3_path: str, **kwargs) -> None:
        fs = s3fs.S3FileSystem(**kwargs)
        with fs.open(s3_path, "w") as file:
            self.to_stream(file)

    @classmethod
    def from_fs(cls, protocol: str, fs_path: str, **kwargs):
        fs = fsspec.filesystem(protocol, **kwargs)
        with fs.open(fs_path, "r") as file:
            return cls.from_stream(file)

    def to_fs(self, protocol: str, fs_path: str, **kwargs) -> None:
        fs = fsspec.filesystem(protocol, **kwargs)
        with fs.open(fs_path, "w") as file:
            self.to_stream(file)


def read_generic(input_file: Union[PATH_TYPE, TextIO], clz: Type[FileIOBase], **kwargs):
    if isinstance(input_file, PATH_TYPE):
        input_file = str(input_file)

        if input_file.startswith("s3://"):
            return clz.from_s3(input_file, **kwargs)
        elif "://" in input_file:
            protocol = input_file.split("://")[0]
            return clz.from_fs(protocol, input_file, **kwargs)
        else:
            return clz.from_file(input_file)
    elif isinstance(input_file, TextIO):
        return FileIOBase.from_stream(input_file)
    else:
        raise ValueError("Invalid input type. Must be str, bytes, pathlib.Path or TextIO")


def write_generic(output_file: Union[PATH_TYPE, TextIO], data: FileIOBase, **kwargs) -> None:
    if isinstance(output_file, PATH_TYPE):
        output_file = str(output_file)
        if output_file.startswith("s3://"):
            data.to_s3(output_file, **kwargs)
        elif "://" in output_file:
            protocol = output_file.split("://")[0]
            data.to_fs(protocol, output_file, **kwargs)
        else:
            data.to_file(output_file)
    elif isinstance(output_file, TextIO):
        data.to_stream(output_file)
    else:
        raise ValueError("Invalid output type. Must be str, bytes, pathlib.Path or TextIO")
