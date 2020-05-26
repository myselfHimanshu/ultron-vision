import json
import pprint
import os
import logging

from logging import Formatter
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

def create_dirs(dirs):
    """
    dirs - list of directories to be created if not exists
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as e:
        logging.getLogger("Dirs Creator.").info(f"Creating directories Error : {e}")
        exit(-1)

def get_config_from_json(json_file):
    """
    Get config from json file
    :param json_file: the path of config file
    :return: config(namespace), config(dictionary)
    """
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = dict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.")
            exit(-1)

def process_config(json_file):
    """
    Get configuration json file
    Processing it with EasyDict to be accessible as attributes then
    editing the paths.
    :param json_file: configuration config file
    :return: config object
    """

    config, _ = get_config_from_json(json_file)
    print("CONFIGURATION OF THIS EXPERIMENT")
    print(config)

    try:
        print("****************************")
        print(f"Experiment : {config['exp_name']}")
        print("****************************")
    except AttributeError:
        print("ERROR!!! Please provide the exp_name")
        exit(-1)

    config["summary_dir"] = os.path.join("experiments", config["exp_name"], "summaries/")
    config["checkpoint_dir"] = os.path.join("experiments", config["exp_name"], "checkpoints/")
    config["out_dir"] = os.path.join("experiments", config["exp_name"], "out/")
    config["log_dir"] = os.path.join("experiments", config["exp_name"], "logs/")
    create_dirs([config["summary_dir"], config["checkpoint_dir"], config["out_dir"], config["log_dir"]])

    # setup logging in the project
    setup_logging(config["log_dir"])

    logging.getLogger().info("Hi, This is Ultron. Nice to meet you!!!")
    logging.getLogger().info("Configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config




    



