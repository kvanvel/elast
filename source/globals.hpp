#ifndef _GLOBALS_H
#define _GLOBALS_H

#include <typeinfo>

#include <fstream>
#include <iostream>
#include <cmath>
//#include

#define PRINT(x) {std::cout << #x <<": "<< x << std::endl;}

namespace stokes{

  typedef double real;

enum myBoundaryID {dir_Minus, dir_Plus, nue_Minus, nue_Plus};

template<class SPARSEMATRIX,class STREAM>
void
PrintMatrixMarket(const SPARSEMATRIX & In,
		  STREAM & out,
		  double threshold
		  )
{
  //Assert( In.m() != 0,  dealii::ExcNotInitialized() );
  //Assert( threshold >0, dealii::ExcMessage("Negative threshold!") );
  out.precision(15);
  // Print the header
  out << "%%MatrixMarket matrix coordinate real general\n";
  const auto nnz = In.n_actually_nonzero_elements(threshold);
  out << In.m() << ' ' << In.n() << ' ' << nnz << '\n';
  
  // Print the body
  for(unsigned int i = 0; i < In.m(); ++i){
    for(auto it = In.begin(i); it != In.end(i); ++it){
      const auto value = it->value();
      if(std::abs(value) >  threshold){
	const unsigned int j = it->column();
	out << i + 1 << ' ' << j + 1 << ' ' << value << '\n';
      }
    }
  }  
}
  

} //end namespace stokes

#endif
