import argparse
import random
import json
from SAM2_FSVOS import SAM2_FSVOS

def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--session_name", type=str, default=str(random.randbytes(4).hex()))
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--group", type=int, default=1)
        parser.add_argument("--test_query_frame_num", type=int, default=None)
        parser.add_argument("--verbose", type=bool, default=False)

        return parser.parse_args()


def main():
    args = get_arguments()
    
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    sam2_predictor = SAM2_FSVOS(
        checkpoint=args.checkpoint,
        config=args.config,
        session_name=args.session_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_query_frame_num=args.test_query_frame_num
    )

    sam2_predictor.test()


if __name__ == '__main__':
    main()