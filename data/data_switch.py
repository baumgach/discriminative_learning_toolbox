
def data_switch(data_identifier):

    # Get Data
    if data_identifier == 'synthetic':
        from data.synthetic_data import synthetic_data as data_loader
    elif data_identifier == 'adni':
        from data.adni_data import adni_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % data_identifier)

    return data_loader