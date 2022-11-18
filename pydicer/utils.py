import os
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import pydicom
import tqdm

from pydicer.config import PyDicerConfig
from pydicer.constants import CONVERTED_DIR_NAME, PYDICER_DIR_NAME

logger = logging.getLogger(__name__)


def hash_uid(uid, truncate=6):
    """Hash a UID and truncate it

    Args:
        uid (str): The UID to hash
        truncate (int, optional): The number of the leading characters to keep. Defaults to 6.

    Returns:
        str: The hashed and trucated UID
    """

    hash_sha = hashlib.sha256()
    hash_sha.update(uid.encode("UTF-8"))
    return hash_sha.hexdigest()[:truncate]


def determine_dcm_datetime(ds, require_time=False):
    """Get a date/time value from a DICOM dataset. Will attempt to pull from SeriesDate/SeriesTime
    field first. Will fallback to StudyDate/StudyTime or InstanceCreationDate/InstanceCreationTime
    if not available.

    Args:
        ds (pydicom.Dataset): DICOM dataset
        require_time (bool): Flag to require the time component along with the date

    Returns:
        datetime: The date/time
    """

    date_type_preference = ["Series", "Study", "InstanceCreation"]

    for date_type in date_type_preference:

        type_date = f"{date_type}Date"
        type_time = f"{date_type}Time"
        if type_date in ds and len(ds[type_date].value) > 0:

            if type_time in ds and len(ds[type_time].value) > 0:
                date_time_str = f"{ds[type_date].value}{ds[type_time].value}"
                if "." in date_time_str:
                    return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")

                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

            if require_time:
                continue

            return datetime.strptime(ds[type_date].value, "%Y%m%d")

    return None


def load_object_metadata(row):
    """Loads the object's metadata

    Args:
        row (pd.Series): The row of the converted DataFrame for which to load the metadata

    Returns:
        pydicom.Dataset: The dataset object containing the original DICOM metadata
    """

    row_path = Path(row.path)

    config = PyDicerConfig()

    # If the working directory is configured and the row_path isn't relative to it, join it.
    if config is not None:

        try:
            row_path.relative_to(config.get_working_dir())
        except ValueError:
            row_path = config.get_working_dir().joinpath(row_path)

    metadata_path = row_path.joinpath("metadata.json")

    if not metadata_path.exists():
        return pydicom.Dataset()
    with open(metadata_path, "r", encoding="utf8") as json_file:
        ds_dict = json.load(json_file)

    return pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)


def load_dvh(row):
    """Loads an object's Dose Volume Histogram (DVH)

    Args:
        row (pd.Series): The row of the converted DataFrame for an RTDOSE

    Raises:
        ValueError: Raised the the object described in the row is not an RTDOSE

    Returns:
        pd.DataFrame: The DataFrame containing the DVH for the row
    """

    if not row.modality == "RTDOSE":
        raise ValueError("Row is not an RTDOSE")

    row_path = Path(row.path)

    config = PyDicerConfig()

    # If the working directory is configured and the row_path isn't relative to it, join it.
    if config is not None:

        try:
            row_path.relative_to(config.get_working_dir())
        except ValueError:
            row_path = config.get_working_dir().joinpath(row_path)

    df_result = pd.DataFrame(columns=["patient", "struct_hash", "label", "cc", "mean"])
    for dvh_file in row_path.glob("dvh_*.csv"):
        col_types = {"patient": str, "struct_hash": str, "label": str, "dose_hash": str}
        df_dvh = pd.read_csv(dvh_file, index_col=0, dtype=col_types)
        df_result = pd.concat([df_result, df_dvh])

    df_result.sort_values(["patient", "struct_hash", "label"], inplace=True)
    df_result.reset_index(drop=True, inplace=True)

    # Change the type of the columns which indicate the dose bins, useful for dose metric
    # computation later
    df_result.columns = [float(c) if "." in c else c for c in df_result.columns]

    return df_result


def read_preprocessed_data(working_directory: Path):
    """Reads the pydicer preprocessed data

    Args:
        working_directory (Path): Working directory for project

    Raises:
        SystemError: Error raised when preprocessed data doesn't yet exist

    Returns:
        pd.DataFrame: The preprocessed data
    """

    pydicer_directory = working_directory.joinpath(PYDICER_DIR_NAME)
    preprocessed_file = pydicer_directory.joinpath("preprocessed.csv")
    if not preprocessed_file.exists():
        raise SystemError("Preprocessed data not found, run preprocess step first")

    # Read the csv
    df_preprocess = pd.read_csv(preprocessed_file, index_col=0)

    # Make sure patient id is a string
    df_preprocess.patient_id = df_preprocess.patient_id.astype(str)

    return df_preprocess


def read_converted_data(
    working_directory: Path,
    dataset_name=CONVERTED_DIR_NAME,
    patients=None,
    join_working_directory=True,
):
    """Read the converted data frame from the supplied data directory.

    Args:
        working_directory (Path): Working directory for project
        dataset_name (str, optional): The name of the dataset for which to extract converted data.
          Defaults to "data".
        patients (list, optional): The list of patients for which to read converted data. If None
            is supplied then all data will be read. Defaults to None.
        join_working_directory (bool, optional): If True, the path in the data frame returned will
            be adjusted to the location of the working_directory. If False the path will be
            relative to the working_directory.

    Returns:
        pd.DataFrame: The DataFrame with the converted data objects.
    """

    dataset_directory = working_directory.joinpath(dataset_name)

    if not dataset_directory.exists():
        raise SystemError(f"Dataset directory {dataset_directory} does not exist")

    df = pd.DataFrame()

    for pat_dir in dataset_directory.glob("*"):

        if not pat_dir.is_dir():
            continue

        pat_id = pat_dir.name

        if patients is not None:
            if pat_id not in patients:
                continue

        # Read in the DataFrame storing the converted data for this patient
        converted_csv = dataset_directory.joinpath(pat_id, "converted.csv")
        if not converted_csv.exists():
            logger.warning("Converted CSV doesn't exist for %s", pat_id)
            continue

        df_converted = pd.read_csv(converted_csv, index_col=0)
        df = pd.concat([df, df_converted])

    # Make sure patient id is a string
    df.patient_id = df.patient_id.astype(str)

    # Join the working directory to each object's path
    if join_working_directory:
        df.path = df.path.apply(lambda p: os.path.join(working_directory, p))

    return df.reset_index(drop=True)


def parse_patient_kwarg(patient):
    """Helper function to prepare patient list from kwarg used in functions throughout pydicer.

    Args:
        patient (list|str): The patient ID or list of patient IDs. If None, all patients in
          dataset_directory are returned.

    Raises:
        ValueError: All patient IDs in list aren't of type `str`
        ValueError: patient was not list, str or None.

    Returns:
        list: The list of patient IDs to process or None if patient is None
    """

    if isinstance(patient, list):
        if not all(isinstance(x, str) for x in patient):
            raise ValueError("All patient IDs must be of type 'str'")
    elif patient is None:
        return None
    else:
        if not isinstance(patient, str) and patient is not None:
            raise ValueError(
                "Patient ID must be list or str. None is a valid to process all patients"
            )
        patient = [patient]

    return patient


def read_simple_itk_image(row):
    """Reads the SimpleITK Image object given a converted dataframe row.

    Args:
        row (pd.Series): The row of the data frame for which to load the SimpleITK Image.

    Returns:
        SimpleITK.Image: The loaded image. Returns None if the image was not found.
    """

    object_path = Path(row.path)

    nifti_path = object_path.joinpath(f"{row.modality}.nii.gz")

    if not nifti_path.exists():
        logger.warning("Unable to load Nifti at %s", nifti_path)
        return None

    return sitk.ReadImage(str(nifti_path))


def get_iterator(iterable, length=None, unit="it", name=None):
    """Get the appropriate iterator based on the level of verbosity configured.

    Args:
        iterable (iterable): The list or iterable to iterate over.
        length (int, optional): The length of the iterator. If None, the len() functio will be used
          to determine the length (only works for list/tuple). Defaults to None.
        unit (str, optional): The unit string to display in the progress bar. Defaults to "it".
        name (str, optional): The name to display in the progress bar. Defaults to None.

    Returns:
        iterable: The appropriate iterable object.
    """

    config = PyDicerConfig()

    iterator = iterable
    if config.get_config("verbosity") == 0:

        if length is None:
            length = len(iterable)

        iterator = tqdm.tqdm(
            iterable,
            total=length,
            unit=unit,
            postfix=name,
        )

    return iterator
