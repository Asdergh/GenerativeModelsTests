import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os
import cv2


class StylesDataHandler:

    def __init__(self
                 , styles_data_dirs
                 , images_data_dir
                 , image_size
                 , reid_lenght=None
                 , gray_mode=False) -> None:
        
        self.gray_mode = gray_mode
        self.styles_data_dirs = styles_data_dirs
        self.images_data_dir = images_data_dir
        self.image_size = image_size
        self.ReId_lenght = reid_lenght
        self.style_naming = None
        self.ReId_files(folder_with_files=self.images_data_dir, id_lenght=self.ReId_lenght)

    def ReId_files(self, folder_with_files, id_lenght):

        if id_lenght is not None:

            id_buffer = []
            for (file_number, file_name) in enumerate(os.listdir(folder_with_files)):
                
                file_path = os.path.join(folder_with_files, file_name)
                while True:

                    curent_id = "".join(rd.choice(["qwertyuiop!@#$%^&"]) for _ in range(id_lenght)) + f"_{file_number}.jpeg"
                    if curent_id not in id_buffer:

                        id_buffer.append(curent_id)
                        dst_file_path = os.path.join(folder_with_files, curent_id)
                        os.rename(file_path, dst_file_path)
                        break

        else:
            pass
                
            
    def rp_image(self, image_path, buffer=None):

        curent_image = cv2.imread(image_path)
        if curent_image is None:
                    
            if len(buffer) != 0:

                curent_image = rd.choice(buffer)
                return curent_image
                  
            else:

                noise = np.random.normal(0, 1, self.image_size)
                return noise
                
        else:
            
            curent_image = cv2.resize(curent_image, self.image_size)
            if self.gray_mode:
                curent_image = cv2.cvtColor(curent_image, cv2.COLOR_BGR2GRAY)
            
            curent_image =  (curent_image / 255.0)
            curent_image_std = (curent_image - np.mean(curent_image)) / np.std(curent_image)
            return curent_image_std


    def _folder_to_collection(self, folder_to_convert, batch_size=None):
        
        images_collection = []
        if batch_size is None:
            images_phs = os.listdir(folder_to_convert)
        
        else:
            
            if batch_size > len(os.listdir(folder_to_convert)):
                batch_size = len(os.listdir(folder_to_convert))
            images_phs = rd.sample(os.listdir(folder_to_convert), batch_size)

        for image_path in images_phs:
            
            image_path = os.path.join(folder_to_convert, image_path)
            preprocessed_image = self.rp_image(image_path=image_path, buffer=images_collection)
            images_collection.append(preprocessed_image)
        
        art_naming = folder_to_convert.split("\\")[-1]

        try:

            images_collection = np.asarray(images_collection)
            return (images_collection, art_naming)
        
        except BaseException:
            return (None, None)


        
    def load_all_styles(self, batch_size=None):
        
        if batch_size is None:
            self.styles_collection = {}

        else:
            styles_collection = {}

        for styles_directory in self.styles_data_dirs:
            
            if not os.path.exists(styles_directory):
                raise FileExistsError(f"cant find styles directory: {styles_directory}!!!")
            
            for styles_folder in os.listdir(styles_directory):
                
                curent_styles_path = os.path.join(styles_directory, styles_folder)
                if not os.path.exists(curent_styles_path):
                    raise FileExistsError(f"cant find style folder: {curent_styles_path} in directory: {styles_directory}")
                
                curent_styles_collection, curent_styles_naming = self._folder_to_collection(curent_styles_path, batch_size=batch_size)
                if curent_styles_collection is not None:

                    if batch_size is None:
                        self.styles_collection[curent_styles_naming] = curent_styles_collection
                    
                    else:
                        styles_collection[curent_styles_naming] = curent_styles_collection
                        
        if batch_size is None:
            return self.styles_collection

        else:
            return styles_collection
        

    def load_styles(self, batch_size=None, style_naming=None):
        
        if self.style_naming is not None:
            style_naming = self.style_naming

        if batch_size is None:
            raise ValueError("can't find requarides parametr batch_size")
        
        if style_naming is not None:
            
            if style_naming in os.listdir(self.styles_data_dirs[0]):
                curent_style_folder = os.path.join(self.styles_data_dirs[0], style_naming)

            elif style_naming in os.listdir(self.styles_data_dirs[1]):
                curent_style_folder = os.path.join(self.styles_data_dirs[1], style_naming)
            
            else:
                raise FileExistsError(f"cant find file: {style_naming}")
        
        else:

            sample_style_dir = rd.choice(self.styles_data_dirs)
            sample_style_naming = rd.choice(os.listdir(sample_style_dir))
            self.style_naming = sample_style_naming
            
            curent_style_folder = os.path.join(sample_style_dir, sample_style_naming)

        styles_tensor = []
        random_images = [rd.choice(os.listdir(curent_style_folder)) for _ in range(batch_size)]
        for image_file in random_images:

            image_path = os.path.join(curent_style_folder, image_file)
            curent_image = self.rp_image(image_path=image_path, buffer=styles_tensor)
            styles_tensor.append(curent_image)
        
        styles_tensor = np.asarray(styles_tensor)
        return styles_tensor

    def load_images(self, batch_size=None):

        if batch_size is None:
            
            images_folder = self.images_data_dir
            self.images_tensor = []
        
        else:
            
            images_folder = rd.sample(os.listdir(self.images_data_dir), batch_size)
            images_tensor = []

        for image_path in images_folder:

            image_path = os.path.join(self.images_data_dir, image_path)

            if batch_size is not None:
                

                preprocessed_image = self.rp_image(image_path=image_path, buffer=images_tensor)
                images_tensor.append(preprocessed_image)

            else:

                preprocessed_image = self.rp_image(image_path=image_path, buffer=images_tensor)
                self.images_tensor.append(preprocessed_image)

        if batch_size is None:

            self.images_tensor = np.asarray(self.images_tensor)
            return self.images_tensor
        
        else:

            images_tensor = np.asarray(images_tensor)
            return images_tensor

  
    def load_batch(self, batch_size, all_styles=False, styles_list=None):

        if all_styles:
        
            styles_batch = self.load_all_styles(batch_size=batch_size)
            images_batch = self.load_images(batch_size=batch_size)
            combination_batch = self.get_combination(number_of_samples=None
                                                    , styles_collection=styles_batch
                                                    , images_tensor=images_batch)
            
            return combination_batch
        
        else:

            if styles_list is not None:
                style_naming = rd.choice(styles_list)

            else:
                style_naming = None

            styles_tensor = self.load_styles(batch_size=batch_size, style_naming=style_naming)
            images_tensor = self.load_images(batch_size=batch_size)

            return (styles_tensor, images_tensor)

        
    def get_combination(self, number_of_samples=None, styles_collection=None
                                   , images_tensor=None):

        if styles_collection is None:
            styles_collection = self.styles_collection
        
        if images_tensor is None:
            images_tensor = self.images_tensor

        combination_collection = {}
        for style in styles_collection.keys():

            if number_of_samples is not None:

                curent_styles_tensor = styles_collection[style][np.random.randint(0, styles_collection[style].shape[0] - 1, number_of_samples)]
                curent_images_tensor = images_tensor[np.random.randint(0, images_tensor.shape[0] - 1, number_of_samples)]
                combinated_samples = [np.asarray(curent_styles_tensor), np.asarray(curent_images_tensor)]
            
            else:
                combinated_samples = [np.asarray(styles_collection[style]), np.asarray(images_tensor)]

            combination_collection[style] = combinated_samples
                
        return combination_collection
    

        


# if __name__ == "__main__":

#     styles_first_dir = "C:\\Users\\1\\Desktop\\GenerativeNeuralNetworkStud\\art_pictures_styles\\dataset\\dataset_updated\\training_set"
#     styles_second_dir = "C:\\Users\\1\\Desktop\\GenerativeNeuralNetworkStud\\van_gogh_styles\\VincentVanGogh"
#     human_faces_dir = "C:\\Users\\1\\Desktop\\GenerativeNeuralNetworkStud\\human_faces\\Faces"

#     sh = StylesDataHandler(image_size=(128, 128), styles_data_dirs=[styles_first_dir, styles_second_dir]
#                         , images_data_dir=human_faces_dir)
    
#     tensors_collection = []
#     tensors_collection.append(sh.load_batch(batch_size=32, all_styles=False, styles_list=["Arles", "Auvers sur Oise", "Sketches in letters"]))
#     tensors_collection.append(sh.load_batch(batch_size=32, all_styles=False))
#     tensors_collection.append(sh.load_batch(batch_size=32, all_styles=False))
#     tensors_collection.append(sh.load_batch(batch_size=32, all_styles=False))

#     fig, axis = plt.subplots(nrows=5, ncols=5)

#     sample_tensors = rd.choice(tensors_collection)
#     sample_number = 0
#     for sample_i in range(axis.shape[0]):
#         for sample_j in range(axis.shape[1]):

#             image = rd.choice(sample_tensors)[sample_number]
#             axis[sample_i, sample_j].imshow(image)
#             sample_number += 1
    
#     plt.show()

   


    






   


        
        
        