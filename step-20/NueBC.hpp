#ifndef _H_NUEBC__
#define _H_NUEBC__

#include "globals.hpp"

namespace elas{

template< int dim>
class NueBC
{
public:
  NueBC (){};

  virtual dealii::SymmetricTensor<2,dim> value(const dealii::Point<dim> &p) const;
  virtual void value_list (const std::vector<dealii::Point<dim> > & points,
			   std::vector<dealii::SymmetricTensor<2,dim> >    & values) const;  
};

} //End Namespace elas

#endif
