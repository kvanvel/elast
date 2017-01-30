#ifndef H_COMPLIANCE__
#define H_COMPLIANCE__

#include "globals.hpp"

namespace elas{

template< int dim>
class Compliance
{
public:
  Compliance (){};

  virtual dealii::SymmetricTensor<4,dim> value(const dealii::Point<dim> &p) const;
  virtual void value_list (const std::vector<dealii::Point<dim> > & points,
			   std::vector<dealii::SymmetricTensor<4,dim> >    & values) const;  
};


} // End namespace elas

#endif
