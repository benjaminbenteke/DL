import argparse
args=argparse.Namespace(
    lr= 1e-2,
    bs= 16,
    train_size= 0.9,
    metadata= './data/metadata_ok.csv',
    path= './data/Images'
)