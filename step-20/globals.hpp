#ifndef H_GLOBALS__
#define H_GLOBALS__

#include <typeinfo>

#include <fstream>
#include <iostream>

#define PRINT(x) {std::cout << #x <<": "<< x << std::endl;}

namespace elas{

typedef double real;  //Interested in changing from double to floats  

enum myBoundaryID {dir_Minus, dir_Plus, nue_Minus, nue_Plus};
}


#endif
