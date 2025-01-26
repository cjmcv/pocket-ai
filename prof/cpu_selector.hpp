/*!
* \brief Choose which CPU core you want to run your code on.
*/

#ifndef POCKET_AI_PROFILER_CPU_SELECTOR_HPP_
#define POCKET_AI_PROFILER_CPU_SELECTOR_HPP_

// refer to: https://github.com/Tencent/ncnn/blob/ee41ef4a378ef662d24f137d97f7f6a57a5b0eba/src/cpu.cpp

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#ifdef __ANDROID__
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#endif

namespace pai {
namespace prof {

class CpuSelector {
public:
    void FetchCpuInfo(bool is_show = false) {
    #ifndef __ANDROID__
        printf("Error: CpuSelector only supports Android system.\n");
        return;
    #endif
        int cpu_count = GetCpuCount();
        printf("cpu numbers %d\n", cpu_count);

        freq_khz_.resize(cpu_count);
        
        for (int i =0 ; i < cpu_count; i++) {
            int max_freq_khz = GetMaxFreqKHz(i);
            freq_khz_[i] = max_freq_khz;
        }

        // distinguish big & little cores with ncnn strategy 
        freq_khz_min_ = INT_MAX;
        freq_khz_max_ = 0;
        for (int i = 0; i < cpu_count; i++) {
            if (freq_khz_[i] > freq_khz_max_)
                freq_khz_max_ = freq_khz_[i];
            if (freq_khz_[i] < freq_khz_min_)
                freq_khz_min_ = freq_khz_[i];
        }
        freq_khz_medium_ = (freq_khz_min_ + freq_khz_max_) / 2;

        if (is_show) {
            for (int i = 0; i < cpu_count; ++i) {
                printf("cpu_%d:%d, ", i, freq_khz_[i]);
            }
            printf("\n");
        }
    }

    void BindCoreWithId(int cpu_id) {
    #ifndef __ANDROID__
        printf("Error: CpuSelector only supports Android system.\n");
        return;
    #endif
        printf("bind cpu_%d:%d \n", cpu_id, freq_khz_[cpu_id]);
        size_t mask = 0;
        mask |= (1 << cpu_id);
        SetSchedAffinity(mask); 
    }

    void BindCoreWithFreq(bool bind_higher_freq) {
    #ifndef __ANDROID__
        printf("Error: CpuSelector only supports Android system.\n");
        return;
    #endif
        printf("bind cpus: ");
        size_t mask = 0;
        if (bind_higher_freq) {    
            for (int i = 0; i < freq_khz_.size(); ++i) {
                if (freq_khz_[i] >= freq_khz_medium_) {
                    mask |= (1 << i);
                    printf("cpu_%d:%d, ", i, freq_khz_[i]);
                }
            }
            
        }
        else {
            for (int i = 0; i < freq_khz_.size(); ++i) {
                if (freq_khz_[i] == freq_khz_min_) {
                    mask |= (1 << i);
                    printf("cpu_%d:%d, ", i, freq_khz_[i]);
                }
            }
        }
        printf("\n");
        SetSchedAffinity(mask); 
    }
private:
    int GetCpuCount() {
        int count = 0;
        FILE* fp = fopen("/proc/cpuinfo", "rb");
        if (!fp)
            return 1;

        char line[1024];
        while (!feof(fp)) {
            char* s = fgets(line, 1024, fp);
            if (!s)
                break;

            if (memcmp(line, "processor", 9) == 0) {
                count++;
            }
        }
        fclose(fp);

        if (count < 1)
            count = 1;

        if (count > (int)sizeof(size_t) * 8) {
            fprintf(stderr, "more than %d cpu detected, thread affinity may not work properly :(\n", (int)sizeof(size_t) * 8);
        }

        return count;
    }

    int GetMaxFreqKHz(int cpuid) {
        // first try, for all possible cpu
        char path[256];
        sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            // second try, for online cpu
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
            fp = fopen(path, "rb");
            if (fp) {
                int max_freq_khz = 0;
                while (!feof(fp)) {
                    int freq_khz = 0;
                    int nscan = fscanf(fp, "%d %*d", &freq_khz);
                    if (nscan != 1)
                        break;

                    if (freq_khz > max_freq_khz)
                        max_freq_khz = freq_khz;
                }
                fclose(fp);
                if (max_freq_khz != 0)
                    return max_freq_khz;
                fp = NULL;
            }

            if (!fp) {
                // third try, for online cpu
                sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
                fp = fopen(path, "rb");

                if (!fp)
                    return -1;

                int max_freq_khz = -1;
                fscanf(fp, "%d", &max_freq_khz);

                fclose(fp);

                return max_freq_khz;
            }
        }

        int max_freq_khz = 0;
        while (!feof(fp)) {
            int freq_khz = 0;
            int nscan = fscanf(fp, "%d %*d", &freq_khz);
            if (nscan != 1)
                break;

            if (freq_khz > max_freq_khz)
                max_freq_khz = freq_khz;
        }
        fclose(fp);
        return max_freq_khz;
    }

    int SetSchedAffinity(size_t thread_affinity_mask) {
    #ifdef __GLIBC__
        pid_t pid = syscall(SYS_gettid);
    #else
    #ifdef PI3
        pid_t pid = getpid();
    #else
        pid_t pid = gettid();
    #endif
    #endif
        int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(thread_affinity_mask), &thread_affinity_mask);
        if (syscallret) {
            fprintf(stderr, "syscall error %d\n", syscallret);
            return -1;
        }
        return 0;
    }

private:
    std::vector<int> freq_khz_;
    int freq_khz_min_;
    int freq_khz_medium_;
    int freq_khz_max_;
};

} // prof.
} // pai.

#endif // POCKET_AI_PROFILER_CPU_SELECTOR_HPP_