#ifndef UTILS_HPP
#define UTILS_HPP
template <typename T> class FlopCounter {
private:
  int mults = 0;
  int adds = 0;

public:
  T mul(T a, T b) {
    mults++;
    return a * b;
  }
  T add(T a, T b) {
    adds++;
    return a + b;
  }
  int get_mults() { return mults; }
};

#endif

