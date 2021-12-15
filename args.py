import argparse 
import logging

def ArgParser():
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--gpu', type=str, default='2', help='select gpu id, -1 is not using')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--print_on_screen', type=bool, default=True, help='print_on_screen')
    parser.add_argument('--model_summary', type=bool, default=True, help='model_summary')
    parser.add_argument('--data_path', type=str, default="./dataset/MapBJ-master/", help='dataset path')
    parser.add_argument('--radius', type=int, default=3, help='hyper parameters: ')
    parser.add_argument('--p', type=int, default=5, help='hyper parameters: ')
    parser.add_argument('--epochs', type=int, default=20, help='hyper parameters: ')
    parser.add_argument('--bs', type=int, default=4096, help='hyper parameters: ')
    parser.add_argument('--maxlen', type=int, default=40, help='hyper parameters: ')
    parser.add_argument('--nworks', type=int, default=6, help='hyper parameters: ')
    args = parser.parse_args()

    logging.info("ArgumentParser:")
    for arg, value in sorted(vars(args).items()):
        logging.info("\tArgument: %s: %r", arg, value)

    return args