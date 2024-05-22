#ifndef POCKET_AI_ENGINE_INFER_PY_MANAGER_HPP_
#define POCKET_AI_ENGINE_INFER_PY_MANAGER_HPP_

#include <Python.h>
#include <frameobject.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <string>
#include <map>
#include <sstream>

namespace pai {
namespace infer {

class PyManager {
public:
    bool Init(std::string work_space, std::string module_name) {
        if (Py_IsInitialized()) {
            printf("Warning: Py_IsInitialized has been called.\n");
        }
        else {
            Py_Initialize();
        }

        if (!Py_IsInitialized()) {
            printf("Error: Py_Initialize failed.\n");
            return false;
        }

        import_array();

        PyRun_SimpleString("import sys");
        std::stringstream ss;
        ss << "sys.path.append('" << work_space << "')";
        printf("%s.\n", ss.str().c_str());
        PyRun_SimpleString(ss.str().c_str());

        // import target module
        module_ = PyImport_ImportModule(module_name.c_str());
        if (!module_) {
            printf("Can not import module: %s \n", module_name.c_str());
            return false;
        }

        // Get method dict from imported module.
        func_dict_ = PyModule_GetDict(module_);
        if (!func_dict_) {
            printf("Error: Can not fine dictionary.\n");
            return false;
        }

        return true;
    }

    void Uninit() {
        Py_XDECREF(module_);
        std::map<std::string, PyObject *>::iterator iter;
        for (iter = class_ins_objs_.begin(); iter != class_ins_objs_.end(); iter++) {
            Py_XDECREF(iter->second);            
        }

        for(uint32_t i=0; i<class_objs_.size(); i++) {
            Py_XDECREF(class_objs_[i]);
        }
        Py_Finalize(); //关闭虚拟机
        printf("Finish Py_Finalize.\n");
    }

    bool Call(std::string func_name, PyObject *input = nullptr, PyObject **ret = nullptr) {
        PyObject *func = PyDict_GetItemString(func_dict_, func_name.c_str());
        if (ret == NULL) {
            PyObject_CallObject(func, input);
        }
        else {
            *ret = PyObject_CallObject(func, input);
        }
        return true;
    }

    // 如有异常，输出异常信息
    void LogException() {
        if (!Py_IsInitialized()) {
            printf("Python 运行环境没有初始化!");
            return;
        }
        
        // 检查错误指示器是否被设置，即检查是否有出现错误
        if (PyErr_Occurred() == NULL) {
            return;
        }

        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);

        if (type) {
            printf("%s ->", PyExceptionClass_Name(type));
        }

        if (value) {
            PyObject *line = PyObject_Str(value);
            if (line && (PyUnicode_Check(line))) {
                printf("%s ", PyUnicode_AsUTF8(line)); // const char* err msg PyUnicode_1BYTE_DATA(line);
            }
            Py_DECREF(line);
        }

        // 在py文件里的错误会有traceback;如果只是cpp端有误，则没有。
        if (traceback) {
            for (PyTracebackObject *tb = (PyTracebackObject *)traceback; tb != NULL; tb = tb->tb_next) {
                PyObject *line = PyUnicode_FromFormat("     File  \"%U\", line %d, in %U\n",
                    tb->tb_frame->f_code->co_filename,
                    tb->tb_lineno,
                    tb->tb_frame->f_code->co_name);
                printf("%s", PyUnicode_1BYTE_DATA(line));
            }
        }
    }

    void CreateClassObj(std::string class_name) {
        PyObject *c_obj = PyObject_GetAttrString(module_,class_name.c_str());
        PyObject *ins = PyObject_CallObject(c_obj, NULL);

        class_ins_objs_[class_name] = ins;
        class_objs_.push_back(c_obj);        
    }

    PyObject *CallClassMethod(std::string class_name, std::string func_name, PyObject *input) {
        std::map<std::string, PyObject*>::iterator iter = class_ins_objs_.find(class_name);
        if (iter != class_ins_objs_.end()) {
            if (input == NULL)
                return PyObject_CallMethod(iter->second, func_name.c_str(), NULL);
            return PyObject_CallMethod(iter->second, func_name.c_str(), "O", input);
        }
        else {
            printf("Error: Can not fine class ins: %s.\n", class_name.c_str());
            return NULL;
        }
    }

public:
    PyObject *module_;
    PyObject *func_dict_;

    std::vector<PyObject *> class_objs_;
    std::map<std::string, PyObject*> class_ins_objs_;
};

} // namespace infer
} // namespace pai

#endif // POCKET_AI_ENGINE_INFER_PY_MANAGER_HPP_