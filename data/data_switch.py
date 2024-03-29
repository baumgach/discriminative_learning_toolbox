
def data_switch(data_identifier):

    # Get Data
    if data_identifier == 'synthetic':
        from data.synthetic_data import synthetic_data as data_loader
    elif data_identifier == 'adni':
        from data.adni_data import adni_data as data_loader
    elif data_identifier == 'acdc':
        from data.acdc_data import acdc_data as data_loader
    elif data_identifier == 'nci_prostate':
        from data.nci_prostate_data import nci_prostate_data as data_loader
    elif data_identifier == 'chestX':
        from data.chestX_data import chestX_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % data_identifier)

    return data_loader