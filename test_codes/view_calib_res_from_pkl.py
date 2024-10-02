import pickle
import sys


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("  please proveid pkl file ..")
        sys.exit()
        
    pkl_file = sys.argv[1]
    
    with open(pkl_file,"rb") as f:
        calib_res = pickle.load(f)
        
    print(calib_res)
    