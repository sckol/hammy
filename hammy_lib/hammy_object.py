from math import e
import re
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from collections import OrderedDict
import xarray as xr


class HammyObject(ABC):
    RESULTS_DIR = Path("results")
    STORAGE = None
    _no_check_metadata = False

    @staticmethod
    def generate_digest(s: str) -> str:
        return hex(abs(int(hashlib.sha256(s.encode()).hexdigest(), 16)))[2:].zfill(6)[
            :6
        ]

    @classmethod
    def get_not_checked_fields(cls) -> list[str]:
        fields = ["id"]
        for base_cls in cls.__mro__:
            if hasattr(base_cls, "_not_checked_fields"):
                fields.extend(base_cls._not_checked_fields)
        return set(fields)

    def __init__(self, id: str | None = None):
        self.resolved = False
        self.metadata = OrderedDict()
        self._id: str | None = id

    def fill_metadata(self, no_check: bool = False) -> None:
        for attr_name, attr_value in self.get_all_variables().items():
            if attr_name in ["metadata", "id", "resolved", "RESULTS_DIR", "STORAGE"]:
                continue
            if isinstance(attr_value, HammyObject):
                for k, v in attr_value.metadata.items():
                    if (
                        k in self.metadata
                        and self.metadata[k] != v
                        and k not in self.get_not_checked_fields()
                    ):
                        if no_check:
                            print(f"[WARNING] Metadata conflict for {k}: {self.metadata[k]} != {v}")
                        else:
                            raise ValueError(
                                f"Metadata conflict for {k}: {self.metadata[k]} != {v}")
                    self.metadata[k] = v
            else:
                self.metadata[attr_name] = self.value_to_clear_string(attr_value)
        new_id = self.id
        if self._id is None:
            self._id = new_id
        elif self._id != new_id:
            if no_check:
                print(f"[WARNING] ID mismatch: {self._id} != {new_id}")
            else:
                raise ValueError(f"ID mismatch: {self._id} != {new_id}")
        self.metadata["id"] = self._id
        self.metadata.move_to_end("id", last=False)

    @staticmethod
    def value_to_clear_string(object: object) -> str:
        str_value = re.sub(r"[\r\n]+", " ", str(object)).strip()
        str_value = re.sub(r"\s+", " ", str_value)
        if len(str_value) > 255:
            str_value = HammyObject.generate_digest(str_value)
        return str_value

    @staticmethod
    def is_regular_field(name, value, type):
        return not (
            name.startswith("_")
            or isinstance(value, (classmethod, staticmethod))
            or callable(value)
            or isinstance(getattr(type, name, None), property)
        )

    def get_all_variables(self) -> dict:
        result = {}
        for cls in reversed(type(self).mro()):
            for name, value in vars(cls).items():
                if self.is_regular_field(name, value, cls):
                    result[name] = value
        instance_attrs = vars(self).items()
        for name, value in instance_attrs:
            if self.is_regular_field(name, value, type(self)):
                result[name] = value
        return result

    def resolve(self, no_load=False, no_check: bool = False) -> None:
        if self.resolved:
            return
        # Iterate over all attributes of the class and resolve all HammyObjects
        for _, attr_value in vars(self).items():
            if isinstance(attr_value, HammyObject):
                attr_value.resolve(no_load=no_load, no_check=no_check)
        self.fill_metadata(no_check=no_check)  # To get the correct id
        if no_load or not self.load(no_check=no_check):
            self.calculate()
        self.fill_metadata(no_check=no_check)
        self.resolved = True

    @property
    def filename(self) -> str:
        folder_name = self.experiment_string or ""
        return self.RESULTS_DIR / folder_name / f"{self.id}.{self.file_extension}"

    @property
    def experiment_string(self) -> str | None:
        found = all(
            f"experiment_{attr}" in self.metadata
            for attr in ["name", "version", "number"]
        )
        if not found:
            return None
        return f"{self.metadata['experiment_number']}_{self.metadata['experiment_name']}_{self.metadata['experiment_version']}"

    def dump(self, no_check: bool = False) -> None:
        self.resolve(no_check=no_check)
        if not self.filename.parent.exists():
            self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.dump_to_filename(self.filename)

    def load(self, no_check: bool = False) -> bool:
        if not self.filename.exists() and self.STORAGE:
            self.STORAGE.download_object(self)
        if not self.filename.exists():
            print(f"Not found the file {self.id}")
            return False
        if not self._no_check_metadata:
            old_metadata = self.metadata.copy()
            self.load_from_filename(self.filename)
            differences = {
                k: (old_metadata.get(k), self.metadata[k])
                for k in set(old_metadata) | set(self.metadata)
                if old_metadata.get(k) != self.metadata.get(k)
            }
            if differences:
                if no_check:
                    print(f"[WARNING] Metadata mismatch for {self.id}: {differences}")
                else:
                    raise ValueError(f"Metadata mismatch for {self.id}: {differences}")
        else:
            self.load_from_filename(self.filename)
        print(f"Loaded cached object {self.id} from file")
        return True

    @abstractmethod
    def dump_to_filename(self, filename: str) -> None:
        pass

    @abstractmethod
    def load_from_filename(self, filename: str) -> None:
        pass

    @abstractmethod
    def calculate(self) -> None:
        pass

    @abstractmethod
    def generate_id(self) -> str:
        pass

    @property
    def id(self) -> str:
        return self._id or self.generate_id()

    @property
    @abstractmethod
    def file_extension(self) -> str:
        pass


class DictHammyObject(HammyObject):
    def dump_to_filename(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load_from_filename(self, filename: str) -> None:
        with open(filename, "r") as f:
            self.metadata = json.load(f, object_pairs_hook=OrderedDict)

    @property
    def file_extension(self) -> str:
        return "json"


class ArrayHammyObject(HammyObject):
    def __init__(self, id=None):
        super().__init__(id)
        self._results: xr.DataArray | xr.Dataset | None = None

    @property
    def results(self) -> xr.DataArray | xr.Dataset | None:
        return self._results

    def dump_to_filename(self, filename: str) -> None:
        self._results.name = "results"
        self._results.attrs.update(
            {k: v for k, v in self.metadata.items() if v is not None}
        )
        encoding = {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32"
            if self._results.dtype == "float64"
            else self._results.dtype,
        }
        self._results.to_netcdf(
            filename, engine="h5netcdf", encoding={"results": encoding}
        )

    def load_from_filename(self, filename: str) -> None:
        with xr.open_dataarray(filename, engine="h5netcdf") as f:
            self._results = f.load()
        self.metadata = OrderedDict(self._results.attrs)

    @property
    def file_extension(self) -> str:
        return "nc"

    @property
    @abstractmethod
    def simple_name(self) -> str:
        pass
