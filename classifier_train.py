# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
from data.data_switch import data_switch

from classifier.model_classifier import classifier

def main():

    # Select experiment below

    # from classifier.experiments import synthetic_CAM as exp_config
    # from classifier.experiments import synthetic_vgg16 as exp_config
    from classifier.experiments import synthetic_resnet34 as exp_config

    # Get Data
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Build VAGAN model
    classifier_model = classifier(exp_config=exp_config, data=data, fixed_batch_size=exp_config.batch_size)

    # Train VAGAN model
    classifier_model.train()


if __name__ == '__main__':

    main()
