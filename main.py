import argparse
import datetime
import logging

import numba as nb

from dlmcol.main_GCP import main_GCP
from dlmcol.main_WVCP import main_WVCP


# Parse arguments
parser = argparse.ArgumentParser(description="DMLCOL for GCP and WVCP")

parser.add_argument("problem",  metavar='t', type=str, help="GCP or WVCP")
parser.add_argument("instance",  metavar='t', type=str, help="instance name")
parser.add_argument("--id_gpu", type=int, help="id_gpu", default=0)
parser.add_argument("--k", type=int, help="number of colors", default=3)
parser.add_argument("--seed", type=int, help="seed", default=0)

parser.add_argument("--alpha", help="alpha", type=float, default=-1)
parser.add_argument("--nb_neighbors", help="nb_neighbors", type=int, default=-1)
parser.add_argument("--nb_iter_tabu", help="nb_iter_tabu", type=int, default=-1)

parser.add_argument('--test', help="test", action='store_true')

args = parser.parse_args()

if args.problem == "GCP":
    name_expe = f"GCP_NN__nb_iter_{args.nb_iter_tabu}_k_{args.k}_{args.instance}_seed_{args.seed}_{datetime.datetime.now()}.txt"
else:
    name_expe = (
        f"DLMCOL_WVCP__{args.instance}_seed_{args.seed}_{datetime.datetime.now()}.txt"
    )

logging.basicConfig(
    handlers=[
        logging.FileHandler(f"logs/{name_expe}.log"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
)


# Init gpu devices
nb.cuda.select_device(args.id_gpu)
device = f"cuda:{args.id_gpu}"
logging.info(device)


if args.problem == "GCP":
    logging.info(f"k : {args.k}")
    main_GCP(
        args.instance,
        args.k,
        args.seed,
        args.alpha if args.alpha != -1 else 0.6,
        args.nb_neighbors if args.nb_neighbors != -1 else 16,
        args.nb_iter_tabu,
        args.test,
        device,
        name_expe,
    )
else:
    main_WVCP(
        args.instance,
        args.seed,
        args.alpha if args.alpha != -1 else 0.2,
        args.nb_neighbors if args.nb_neighbors != -1 else 32,
        args.nb_iter_tabu,
        args.test,
        device,
        name_expe,
    )
