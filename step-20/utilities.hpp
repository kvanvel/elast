#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include "globals.hpp"

template<class SPARSEMATRIX>
void
DistillMatrix(SPARSEMATRIX & matrixIN,
	      typename dealii::SparsityPattern & sparsityPattern)
	      
{
  
  Assert( matrixIN.m() != 0, dealii::ExcNotInitialized() );  

  
  const dealii::SparseMatrix<elas::real>::size_type n_rows = matrixIN.m();
  const dealii::SparseMatrix<elas::real>::size_type n_cols = matrixIN.n();
  
  std::vector<std::vector<dealii::SparseMatrix<elas::real>::size_type > > col_indices(n_rows );
  std::vector<std::vector<std::pair<dealii::SparseMatrix<elas::real>::size_type, elas::real> > >
    colVal_indices(n_rows );

  elas::real zero = 0.0;
  
  for(unsigned int i = 0; i < n_rows; ++i){    
    for(auto it = matrixIN.begin(i); it != matrixIN.end(i); ++it){
      const auto value = it->value();      
      if(value != zero){
	//  We don't play games here with trying
	//  to determine if a small number should be interpreted as zero.  We only remove exact zeros.
	const auto col = it->column();
	const std::pair<dealii::SparseMatrix<elas::real>::size_type, elas::real> tempPair {col, value};
	col_indices[i].emplace_back(col);
	colVal_indices[i].emplace_back(tempPair);
      }
    }
  }
  //const bool optimizeDiagonal = true;
  sparsityPattern.copy_from(n_rows,
			    n_cols,
			    colVal_indices.begin(), 
			    colVal_indices.end()
			    //optimizeDiagonal
			    );
  
  matrixIN.clear();
  matrixIN.reinit(sparsityPattern);
  matrixIN.copy_from(colVal_indices.begin(),
		     colVal_indices.end() );
  
  sparsityPattern.compress();
}


template<class SPARSEMATRIX,class STREAM>
void
PrintMatrixMarket(const SPARSEMATRIX & In,
		  STREAM & out,
		  double threshold = 1e-10);
		  

template<class SPARSEMATRIX,class STREAM>
void
PrintMatrixMarket(const SPARSEMATRIX & In,
		  STREAM & out,
		  double threshold
		  )
{
  Assert( In.m() != 0,  dealii::ExcNotInitialized() );
  Assert( threshold >0, dealii::ExcMessage("Negative threshold!") );
  out.precision(15);
  // Print the header
  out << "%%MatrixMarket matrix coordinate real general\n";
  const auto nnz = In.n_actually_nonzero_elements(threshold);
  out << In.m() << ' ' << In.n() << ' ' << nnz << '\n';
  
  // Print the body
  for(unsigned int i = 0; i < In.m(); ++i){
    for(auto it = In.begin(i); it != In.end(i); ++it){
      const auto value = it->value();
      if(fabs(value) >  threshold){
	const unsigned int j = it->column();
	out << i + 1 << ' ' << j + 1 << ' ' << value << '\n';
      }
    }
  }  
}



#endif
