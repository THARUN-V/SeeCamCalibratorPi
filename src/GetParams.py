import json

class GetParams:
    
    def __init__(self,file_path):
        
        try:
            with open(file_path,"r") as json_file:
                self.param_file = json.load(json_file)
        except FileNotFoundError:
            print(f"------- {file_path} Not Found ---------")
            exit()
            
        self.camera_resolution = self.param_file["camera"]["resolution"]            