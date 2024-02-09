#include <iostream>
#include <cmath>
#include <vector>


#ifdef FLOAT_TYPE
#define TYPE float
#else
#define TYPE double
#endif


int main(){
    std::vector<float> vec;
    TYPE step = powl(10, 7);



    for (TYPE i = 0; i < step; i++){
        vec.push_back(sin(2 * M_PI * i / step));
    }

    TYPE sum;
    for (TYPE num : vec){
        sum += num;
    }

    std::cout << sum;

    return 0;
}