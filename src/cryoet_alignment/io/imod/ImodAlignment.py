import os
from typing import Optional, Union

from pydantic import BaseModel

from cryoet_alignment.io.imod.newst import ImodNEWSTCOM
from cryoet_alignment.io.imod.rawtlt import ImodTLT, ImodXTILT
from cryoet_alignment.io.imod.tilt import ImodTILTCOM
from cryoet_alignment.io.imod.xf import ImodXF

PATH_TYPE = Union[str, bytes, os.PathLike]


class ImodAlignment(BaseModel):
    xf: ImodXF
    tlt: ImodTLT
    xtilt: Optional[ImodXTILT] = None
    tiltcom: Optional[ImodTILTCOM] = None
    newstcom: Optional[ImodNEWSTCOM] = None

    @classmethod
    def read(
        cls,
        base_name: PATH_TYPE = None,
        xf_path: PATH_TYPE = None,
        tlt_path: PATH_TYPE = None,
        xtilt_path: PATH_TYPE = None,
        tiltcom_path: PATH_TYPE = None,
        newstcom_path: PATH_TYPE = None,
    ):
        if not base_name and not all([xf_path, tlt_path]):
            raise ValueError("Either base_name or xf_path and tlt_path must be provided.")

        if base_name and any([xf_path, tlt_path, xtilt_path, tiltcom_path, newstcom_path]):
            raise ValueError("Either base_name or xf_path and tlt_path must be provided, not both.")

        if base_name:
            xf = ImodXF.from_file(f"{base_name}.xf")
            tlt = ImodTLT.from_file(f"{base_name}.tlt")
            xtilt = ImodXTILT.from_file(f"{base_name}.xtilt") if os.path.exists(f"{base_name}.xtilt") else None
            parent = os.path.dirname(base_name)
            tiltcom = ImodTILTCOM.from_file(f"{parent}/tilt.com") if os.path.exists(f"{parent}/tilt.com") else None
            newstcom = ImodNEWSTCOM.from_file(f"{parent}/newst.com") if os.path.exists(f"{parent}/newst.com") else None
        else:
            xf = ImodXF.from_file(xf_path)
            tlt = ImodTLT.from_file(tlt_path)
            xtilt = ImodXTILT.from_file(xtilt_path) if xtilt_path else None
            tiltcom = ImodTILTCOM.from_file(tiltcom_path) if tiltcom_path else None
            newstcom = ImodNEWSTCOM.from_file(newstcom_path) if newstcom_path else None

        return cls(xf=xf, tlt=tlt, xtilt=xtilt, tiltcom=tiltcom, newstcom=newstcom)

    def write(
        self,
        base_name: PATH_TYPE = None,
        xf_path: PATH_TYPE = None,
        tlt_path: PATH_TYPE = None,
        xtilt_path: PATH_TYPE = None,
        tiltcom_path: PATH_TYPE = None,
        newstcom_path: PATH_TYPE = None,
    ):
        if not base_name and not all([xf_path, tlt_path]):
            raise ValueError("Either base_name or xf_path and tlt_path must be provided.")

        if base_name and any([xf_path, tlt_path, xtilt_path, tiltcom_path, newstcom_path]):
            raise ValueError("Either base_name or xf_path and tlt_path must be provided, not both.")

        if base_name:
            self.xf.to_file(f"{base_name}.xf")
            self.tlt.to_file(f"{base_name}.tlt")
            if self.xtilt:
                self.xtilt.to_file(f"{base_name}.xtilt")
            if self.tiltcom:
                parent = os.path.dirname(base_name)
                self.tiltcom.to_file(f"{parent}/tilt.com")
            if self.newstcom:
                parent = os.path.dirname(base_name)
                self.newstcom.to_file(f"{parent}/newst.com")
        else:
            self.xf.to_file(xf_path)
            self.tlt.to_file(tlt_path)
            if self.xtilt:
                self.xtilt.to_file(xtilt_path)
            if self.tiltcom:
                self.tiltcom.to_file(tiltcom_path)
            if self.newstcom:
                self.newstcom.to_file(newstcom_path)
