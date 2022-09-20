import os
import datetime


def check_images_download(file_list):
    """
    Check if images from the file_list in db entry have been downloaded
    to the path in local datastore
    Args:
        file_list: paths of the files that need to validated

    Returns:
        True - If the images are present in the path
        failure_message - If any of the images have not been downloaded properly.

    """
    for file in file_list:
        image_path = file['path']
        if not os.path.exists(image_path):
            failure_message = "Image {} does not exist on disk".format(image_path)
            return False, failure_message
    return True, "success"


def set_up_debug(debug_info, analysis_entry, logger, mount_path=None):
    def set_up_local_files(aws_bucket, env, analysis_entry):
        for index, entry in enumerate(analysis_entry['input']['file_list']):
            file_path = entry['path']

            if mount_path:
                local_path = os.path.join(mount_path, file_path.lstrip("/"))
            else:
                local_path = file_path
            #
            # if debug_info['mac_os']:
            #     local_path = append_home_path(local_path)
            # else:
            #     local_path = file_path

            if os.path.exists(local_path):
                analysis_entry['input']['file_list'][index]['path'] = local_path
                continue
            logger.info("Downloding {} to {}".format(file_path, local_path))
            command = "gsutil cp gs://{}/{} {}".format(aws_bucket, os.path.join(env, file_path.lstrip('/')), local_path)
            print(command)
            os.system(command)
            analysis_entry['input']['file_list'][index]['path'] = local_path

        ### saving the wbc extraction output
        if "device_wbc_ex_status" in analysis_entry['input'].keys():
            if analysis_entry['input']["device_wbc_ex_status"]:
                fil_path_json = analysis_entry['input']["patches_json_path"]
                if mount_path:
                    local_path_json = os.path.join(mount_path, fil_path_json.lstrip("/"))
                else:
                    local_path_json = file_path

                logger.info("Downloding wbc output file {} to {}".format(fil_path_json, local_path_json))
                command = "gsutil cp gs://{}/{} {}".format(aws_bucket, os.path.join(env, fil_path_json.lstrip('/')),
                                                           local_path_json)
                print(command)
                os.system(command)
                analysis_entry['input']['patches_json_path'] = os.path.abspath(local_path_json)
        return analysis_entry

    aws_bucket = debug_info['aws_bucket']
    env = debug_info['env']
    if debug_info['download_files']:
        return set_up_local_files(aws_bucket, env, analysis_entry)
    return analysis_entry


def append_home_path(path):
    from os.path import expanduser
    home = expanduser("~")
    # Using lstrip to remove the leading "/" python does not prepend absolute paths.
    # https://docs.python.org/3/library/os.path.html#os.path.join
    return os.path.join(home, path.lstrip('/'))


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_time():
    """
    To ensure that all the calls to set time through a common interface this is being used.
    Returns:

    """
    time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    return time


def get_time_diff(end_date, start_date):
    def __datetime(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')

    start = __datetime(start_date)
    end = __datetime(end_date)
    delta = end - start
    return delta

