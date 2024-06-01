#ifndef POCKET_AI_ENGINE_INFER_PY_TFLITE_CPP_PY_HPP_
#define POCKET_AI_ENGINE_INFER_PY_TFLITE_CPP_PY_HPP_

#include "py_manager.hpp"

#include <vector>
#include <string>
#include <map>

namespace pai {
namespace infer {

struct FTensor {
    std::string name;
    int dims;
    int shape[6] = {0};
    int type_size;
    int size;
    void *data;
};

class TfliteCpy {
public:
    void Init(std::string work_space, std::string model_name) {

        std::string module_name = "tflite_inference";
        class_name_ = "TfliteInference";

        pm_ = new PyManager();
        pm_->Init(work_space, module_name.c_str());

        pm_->CreateClassObj(class_name_.c_str());
        pm_->CallClassMethod(class_name_.c_str(), "hello", NULL);

        PyObject *args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, Py_BuildValue("s", model_name.c_str()));
        pm_->CallClassMethod(class_name_.c_str(), "load_model", args);

        Py_DECREF(args);

        CreateIo();
        printf("Finish Init.\n");
    }

    void Unitit() {
        for (uint32_t i=0; i<inputs_.size(); i++) {
            FTensor *tensor = inputs_[i];
            free(tensor->data);
            delete tensor;
        }
        for (uint32_t i=0; i<outputs_.size(); i++) {
            FTensor *tensor = outputs_[i];
            free(tensor->data);
            delete tensor;
        }
        pm_->Uninit();
        delete pm_;
    }

    bool GetInputPtr(std::string input_name, void **data, uint32_t *size) {
        for (uint32_t i=0; i<inputs_.size(); i++) {
            FTensor *tensor = inputs_[i];
            if (tensor->name == input_name) {
                *data = tensor->data;
                *size = tensor->size;
                printf("input found.\n");
                return true;
            }
        }

        printf("Error: Can nopt fine input: %s.\n", input_name.c_str());
        return false;
    }

    bool GetOutputPtr(std::string output_name, void **data, uint32_t *size) {
        for (uint32_t i=0; i<outputs_.size(); i++) {
            FTensor *tensor = outputs_[i];
            if (tensor->name == output_name) {
                *data = tensor->data;
                *size = tensor->size;
                printf("output found.\n");
                return true;
            }
        }

        printf("Error: Can nopt fine output: %s.\n", output_name.c_str());
        return false;
    }

    void *GetNpyData(std::string data_path) {
        PyObject *args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, Py_BuildValue("s", data_path.c_str()));
        PyObject *ret_obj = pm_->CallClassMethod(class_name_.c_str(), "get_npy", args);

        PyArrayObject *arr_obj;
        PyArg_Parse(ret_obj, "O", &arr_obj);

        void *data = (void *)PyArray_DATA(arr_obj);
        return data;
    }

    void Infer() {
        for (uint32_t i=0; i<inputs_.size(); i++) {
            PyObject *args = PyTuple_New(2);

            FTensor *tensor = inputs_[i];
            npy_intp shape[6] = {tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3], tensor->shape[4], tensor->shape[5]};
            PyTuple_SetItem(args, 0, Py_BuildValue("i", i));

            int typenum = NPY_FLOAT;
            if (tensor->type_size == 1)
                typenum = NPY_INT8;
            PyTuple_SetItem(args, 1, PyArray_SimpleNewFromData(tensor->dims, shape, typenum, tensor->data));
            pm_->CallClassMethod(class_name_.c_str(), "set_input", args);

            Py_DECREF(args);
        }

        pm_->CallClassMethod(class_name_.c_str(), "run", NULL);
        pm_->LogException();

        PyObject *ret_obj;
        for (uint32_t i=0; i<outputs_.size(); i++) {
            PyObject *args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, Py_BuildValue("i", i));
            ret_obj = pm_->CallClassMethod(class_name_.c_str(), "get_output", args);
            Py_DECREF(args);

            PyArrayObject *arr_obj;
            PyArg_Parse(ret_obj, "O", &arr_obj);
          
            void *data = (void *)PyArray_DATA(arr_obj);
            memcpy(outputs_[i]->data, data, outputs_[i]->size);
        }
    }

    void Print(std::string tensor_name) {

        std::string tensor_attri = "";
        FTensor *target = NULL;
        for (uint32_t i=0; i<inputs_.size(); i++) {
            FTensor *tensor = inputs_[i];
            if (tensor->name == tensor_name) {
                target = tensor;
                tensor_attri = "Input Tensor";
                break;
            }
        }
        if (target == NULL) {
            for (uint32_t i=0; i<outputs_.size(); i++) {
                FTensor *tensor = outputs_[i];
                if (tensor->name == tensor_name) {
                    target = tensor;
                    tensor_attri = "Output Tensor";
                    break;
                }
            }
        }

        if (target == NULL) {
            printf("Error: Can not find tensor: %s\n", tensor_name.c_str());
            std::abort();
        }

        printf("%s(%s): \n", tensor_attri.c_str(), tensor_name.c_str());
        for (uint32_t i=0; i<target->size/target->type_size; i++) {
            if (target->type_size == 1)
                printf("%d", ((int8_t *)target->data)[i]);
            else
                printf("%f", ((float_t *)target->data)[i]);

            if (i < target->size/target->type_size-1)
                printf(", ");    
        }
        printf("\n");
    }

private:
    void CreateIo() {
        PyObject *ret_obj;
        ret_obj = pm_->CallClassMethod(class_name_.c_str(), "get_io_num", NULL);
        int input_num, output_num;
        PyArg_ParseTuple(ret_obj, "ii", &input_num, &output_num);

        PyObject *args = PyTuple_New(1);
        PyArrayObject *shape_obj;
        char *name;
        int type_size;
        for (uint32_t i=0; i<input_num; i++) {
            PyTuple_SetItem(args, 0, Py_BuildValue("i", i));
            ret_obj = pm_->CallClassMethod(class_name_.c_str(), "get_inputs_detail", args);
            PyArg_ParseTuple(ret_obj, "isO", &type_size, &name , &shape_obj);
            inputs_.push_back(CreateTensor(type_size, name, shape_obj));
        }
        for (uint32_t i=0; i<output_num; i++) {
            PyTuple_SetItem(args, 0, Py_BuildValue("i", i));
            ret_obj = pm_->CallClassMethod(class_name_.c_str(), "get_outputs_detail", args);
            PyArg_ParseTuple(ret_obj, "isO", &type_size, &name , &shape_obj);
            outputs_.push_back(CreateTensor(type_size, name, shape_obj));
        }
        Py_DECREF(args);

        for (size_t i=0; i<inputs_.size(); i++) {
            FTensor *tensor = inputs_[i];
            printf("in %s: type_size(%d), dims(%d), [%d, %d, %d, %d, %d, %d]\n", 
                tensor->name.c_str(), tensor->type_size, tensor->dims, 
                tensor->shape[0], tensor->shape[1], tensor->shape[2], 
                tensor->shape[3], tensor->shape[4], tensor->shape[5]);
        }

        for (size_t i=0; i<outputs_.size(); i++) {
            FTensor *tensor = outputs_[i];
            printf("out %s: type_size(%d), dims(%d), [%d, %d, %d, %d, %d, %d]\n", 
                tensor->name.c_str(), tensor->type_size, tensor->dims, 
                tensor->shape[0], tensor->shape[1], tensor->shape[2], 
                tensor->shape[3], tensor->shape[4], tensor->shape[5]);
        }
    }

    FTensor *CreateTensor(int type_size, std::string name, PyArrayObject *shape_obj) {
        FTensor *t = new FTensor();

        int *data = (int *)PyArray_DATA(shape_obj);
        t->dims = PyArray_DIM(shape_obj, 0);

        t->size = 1;
        for (uint32_t i=0; i<t->dims; i++) {
            t->shape[i] = data[i];
            t->size *= data[i];
        }
        t->type_size = type_size;
        t->size *= type_size;
        t->data = (void *)malloc(t->size);
        t->name = name;
        return t;
    }

private:
    PyManager *pm_;
    std::string class_name_;

    std::vector<FTensor*> inputs_;
    std::vector<FTensor*> outputs_;
};

// // Demo
// #include <stdio.h>
// #include "tflite_cpy.hpp"

// int main() {
//     pai::infer::TfliteCpy tflite_cpy;
//     tflite_cpy.Init("tflite_cpy", "TfliteInference", "../models/tf_micro_conv_test_model.int8.tflite");

//     int8_t *input_data;
//     uint32_t input_size;
//     tflite_cpy.GetInputPtr("serving_default_conv2d_input:0", (void **)&input_data, &input_size);

//     for (uint32_t i=0; i<input_size/sizeof(uint8_t); i++)
//         input_data[i] = i % 255;

//     tflite_cpy.Infer();

//     tflite_cpy.Print("StatefulPartitionedCall:0");

//     return 0;
// }

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFER_PY_TFLITE_CPP_PY_HPP_