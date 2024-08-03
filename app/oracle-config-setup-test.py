# Please check https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm
# for help on how to generate a key-pair and calculate the key fingerprint.

from oci.config import from_file, validate_config

config = from_file(file_location= "~/.oci/config") # pulls DEFAULT config

validate_config(config)
