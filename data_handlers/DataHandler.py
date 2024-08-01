import numpy as np
import os
import json as js
import cv2
import random as rd


class DataHandler():

    def __init__(self, data_dir, image_size) -> None:
        

        self.data_dir = data_dir
        self.class_types = {
            "glioma": 0, 
            "meningioma": 1
        }
        self.image_size = image_size
    

    def _add_detection_sample(self, glioma, meningioma, notumor, train_or_test):

        if train_or_test == "Training":

            if rd.randint(0, 100) > 50.0:
                self.train_detector_dataset.append([glioma, 0])
            
            else:
                self.train_detector_dataset.append([meningioma, 0])
            
            self.train_detector_dataset.append([notumor, 1])

        elif train_or_test == "Testing":

            if rd.randint(0, 100) > 50.0:
                self.test_detector_dataset.append([glioma, 0])
            
            else:
                self.test_detector_dataset.append([meningioma, 0])
            
            self.test_detector_dataset.append([notumor, 1])

        else:
            raise ValueError(f"value 'train_or_test' cant't be set with: {train_or_test}")


    def _add_separation_sample(self, glioma, meningioma, train_or_test):

        if train_or_test == "Training":
            
            self.train_separator_dataset.append([glioma, self.class_types["glioma"]])
            self.train_separator_dataset.append([meningioma, self.class_types["meningioma"]])

        elif train_or_test == "Testing":
            
            self.test_separator_dataset.append([glioma, self.class_types["glioma"]])
            self.test_separator_dataset.append([meningioma, self.class_types["meningioma"]])
        
        else:
            raise ValueError(f"value 'train_or_test' cant't be set with: {train_or_test}")


    def _ret_test(self, sample):

        result_sample = sample
        if sample is None:
            result_sample = np.random.normal(0, 1.23, self.image_size)
            result_sample = (result_sample - np.mean(result_sample)) / np.std(result_sample)
        
        else:
            result_sample = cv2.resize(result_sample, self.image_size)
            result_sample = (result_sample - np.mean(result_sample)) / np.std(result_sample)

        return result_sample
        

    def load_data(self):
        
        self.train_detector_dataset = []
        self.train_separator_dataset = []
        self.test_detector_dataset = []
        self.test_separator_dataset = []

        for data_batch_path in os.listdir(self.data_dir):
            
            glioma_dir = os.path.join(self.data_dir, data_batch_path, "glioma")
            meningioma_dir = os.path.join(self.data_dir, data_batch_path, "meningioma")
            notumor_dir = os.path.join(self.data_dir, data_batch_path, "notumor")

            for (glioma_sample_path, 
                 meningioma_sample_path, 
                 notumor_sample_path) in zip(os.listdir(glioma_dir),
                                            os.listdir(meningioma_dir),
                                            os.listdir(notumor_dir)):
                
                glioma_sample_path = os.path.join(glioma_dir, glioma_sample_path)
                meningioma_sample_path = os.path.join(meningioma_dir, meningioma_sample_path)
                notumor_sample_path = os.path.join(notumor_dir, notumor_sample_path)

                glioma_sample = self._ret_test(cv2.imread(glioma_sample_path))
                meningioma_sample = self._ret_test(cv2.imread(meningioma_sample_path))
                notumor_sample = self._ret_test(cv2.imread(notumor_sample_path))

                self._add_detection_sample(glioma=glioma_sample,
                                           meningioma=meningioma_sample,
                                           notumor=notumor_sample,
                                           train_or_test=data_batch_path)
                
                self._add_separation_sample(glioma=glioma_sample,
                                            meningioma=meningioma_sample,
                                            train_or_test=data_batch_path)

        return (self.train_detector_dataset, self.test_detector_dataset), (self.train_separator_dataset, self.test_separator_dataset)

    def permutate_data(self):

        rd.shuffle(self.train_detector_dataset)
        rd.shuffle(self.test_detector_dataset)
        rd.shuffle(self.train_separator_dataset)
        rd.shuffle(self.test_separator_dataset)

        return (self.train_detector_dataset, self.test_detector_dataset), (self.train_separator_dataset, self.test_separator_dataset)

    def _trashhold_sample(self, sample):

        sample_copy = sample

        sample_spector_0 = sample_copy[:, :, 0]
        sample_spector_1 = sample_copy[:, :, 1]
        sample_spector_2 = sample_copy[:, :, 2]

        sample_spector_0[sample_spector_0 < np.mean(sample_spector_0)] = 0.0
        sample_spector_1[sample_spector_1 < np.mean(sample_spector_1)] = 0.0
        sample_spector_2[sample_spector_2 < np.mean(sample_spector_2)] = 0.0

        sample_copy[:, :, 0] = sample_spector_0
        sample_copy[:, :, 1] = sample_spector_1
        sample_copy[:, :, 2] = sample_spector_2

        sample_copy_std = (sample_copy - np.mean(sample_copy)) / np.std(sample_copy)
        return sample_copy_std

    def trashhold_data(self):

   
        for (sample_number, (train_separator_sample,
            test_separator_sample,
            train_detector_sample,
            test_detector_sample)) in enumerate(zip(self.train_separator_dataset,
                                        self.test_separator_dataset,
                                        self.train_detector_dataset,
                                        self.test_detector_dataset)):
            
            trashholeded_train_sep = self._trashhold_sample(train_separator_sample[0])
            trashholeded_test_sep = self._trashhold_sample(test_separator_sample[0])
            trashholeded_train_det = self._trashhold_sample(train_detector_sample[0])
            trashholeded_test_det = self._trashhold_sample(test_detector_sample[0])

            self.train_detector_dataset[sample_number][0] = trashholeded_train_sep
            self.test_detector_dataset[sample_number][0] = trashholeded_test_sep
            self.train_separator_dataset[sample_number][0] = trashholeded_train_det
            self.test_separator_dataset[sample_number][0] = trashholeded_test_det
        
        return (self.train_detector_dataset, self.test_detector_dataset), (self.train_separator_dataset, self.test_separator_dataset)


    def generate_random_shapes(self, shapes_number):

        self.shapes_datatensor = np.zeros(shape=(shapes_number, self.image_size[0], self.image_size[1], self.image_size[0]))
        for sample_number in range(shapes_number):
            
            random_idx = np.random.randint(0, len(self.train_detector_dataset), self.image_size[0])
            volume_shape = np.asarray([self.train_detector_dataset[idx][0][:, :, np.random.randint(0, 2)] for idx in random_idx])
            self.shapes_datatensor[sample_number, :, :, :] = volume_shape

        return self.shapes_datatensor
           




                
                
                

                








        
        
    
    
    

        
        

                




        

    


                