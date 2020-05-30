import argparse

from agents import *
from utils.config import process_config

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config', metavar='config_json_file',
        default=None, help="Configuration file in JSON format."
    )

    args = arg_parser.parse_args()

    # parse config file
    config = process_config(args.config)

    # Create the agent and pass all the configuration to it and run
    agent_class = globals()[config["agent"]]
    agent = agent_class(config)
    
    # train model
    agent.run()
    agent.finalize()
    
    # visualize plots
    agent.plot_accuracy_graph()
    agent.plot_loss_graph()
    agent.show_misclassified_images()

    # predict and show grad cam images
    # agent.predict()


if __name__=="__main__":
    main()