#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

class Timer {
private:
  std::unordered_map<std::string, float> func_timer;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::string current_func = "";

public:
  void start_timer(std::string func_name) {
    if (current_func != "") {
      std::cerr << "Didn't end previous timer " << current_func
                << " before starting a new one" << std::endl;
      exit(1);
    }
    current_func = func_name;
    start = std::chrono::high_resolution_clock::now();
  }
  void end_timer(std::string func_name) {
    if (current_func != func_name) {
      std::cerr << "Trying to end previous timer " << current_func
                << " using a different one " << func_name << std::endl;
      exit(1);
    }
    end = std::chrono::high_resolution_clock::now();
    float this_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    current_func = "";
    auto it = func_timer.find(func_name);
    if (it != func_timer.end())
      it->second += this_time;
    else
      func_timer[func_name] = this_time;
  }
  void print_all_times() {
    for (auto &p : func_timer) {
      std::cout << p.first << ", " << p.second << std::endl;
    }
  }
};
#endif
