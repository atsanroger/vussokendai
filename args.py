##
##        @file    args.py
##        @brief
##        @author  Wei-Lun Chen (wlchen)
##                 $LastChangedBy: wlchen $
##        @date    $LastChangedDate: 2024-10-23 16:42:21 #$
##        @version $LastChangedRevision: 2499 $
##

from my_header import argparse

def parse_Q0(value):
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid boolean or float")


parser = argparse.ArgumentParser(description='OMG.')
parser.add_argument(
    '--Fit', action='store_true',
    help='Set fitting to True if provided, otherwise False'
                )
parser.add_argument(
    '--Debug', action='store_true', 
    help='Open the debug mode if provided, otherwise False'
    )
parser.add_argument(
    '--SavePic', action='store_true', 
    help='Save Picture'
    )
parser.add_argument(
    '--EO', type= str, default= None, 
    help='Even-odd, Even and Odd or None for usual'
    )
parser.add_argument(
    '--t0', type=float, default = 0.5, 
    help='Time slicing offsetm, with 2time value'
    )
parser.add_argument(
    '--ND', 
    type = int, 
    default = 1, 
    help = 'Set ratio of ND to use, default is one'
    )
parser.add_argument(
    '--T', type=int, default=12, 
    help = 'TimeSlicing'
    )
parser.add_argument(
    '--Q0', 
    type = parse_Q0, 
    default = True, 
    help = 'True, False, or float'
    )


args= parser.parse_args()

Debug  = args.Debug
Fit    = args.Fit
Save   = args.SavePic

t0     = args.t0
ND     = args.ND
EO     = args.EO
Q0     = args.Q0
T      = args.T