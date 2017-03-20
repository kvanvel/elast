#include "compliance.hpp"
#include "DirBC.hpp"
#include "NueBC.hpp"
#include "elasRHS.hpp"


namespace elas{
namespace LocalIntegrators{
namespace LDG{

template <int dim>
void
massSigma
( dealii::FullMatrix<elas::real> & tau_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const elas::real factor)
{  
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  AssertDimension(tau_in_sigma_in.m(), n_sigma_dofs);
  AssertDimension(tau_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim ) /2 );

  std::vector<dealii::SymmetricTensor<4,dim > >
    compliance_values (fe_sigma_Self.n_quadrature_points);
  elas::Compliance<dim> compliance;

  compliance.value_list(fe_sigma_Self.get_quadrature_points(),
  			compliance_values);  
  
  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const auto & JxW = fe_sigma_Self.get_JxW_values();
  

  for(unsigned int point = 0;
      point < fe_sigma_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_sigma_dofs; ++i){
      const dealii::SymmetricTensor<2,dim> tau_i = fe_sigma_Self[stress].value(i,point);
      const auto tempterm = compliance_values[point] * tau_i;      
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
  	const dealii::SymmetricTensor<2,dim> sigma_j = fe_sigma_Self[stress].value(j,point);
  	const auto tempterm2 = (tempterm * sigma_j) * JxW[point]* factor;
  	tau_in_sigma_in(i,j) += tempterm2;	
      }
    }
  }
}


template <int dim>
void
massSigma_NEW
( dealii::FullMatrix<elas::real> & tau_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const elas::real factor)
  
{  
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  AssertDimension(tau_in_sigma_in.m(), n_sigma_dofs);
  AssertDimension(tau_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim ) /2 );

  std::vector<dealii::SymmetricTensor<4,dim > > compliance_values (fe_sigma_Self.n_quadrature_points);
  elas::Compliance<dim> compliance;
  
  compliance.value_list(fe_sigma_Self.get_quadrature_points(),
  			compliance_values);  
  
  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const auto & JxW = fe_sigma_Self.get_JxW_values();
  dealii::SymmetricTensor<2,dim> tau_i;
  dealii::SymmetricTensor<2,dim> sigma_j;
  dealii::SymmetricTensor<2,dim> complianceDot_tau_i;
  elas::real doubledotProduct;
  elas::real JxW_at_point;
  

  for(unsigned int point = 0;
      point < fe_sigma_Self.n_quadrature_points;
      ++point){
    JxW_at_point = JxW[point];
    for(unsigned int i = 0; i < n_sigma_dofs; ++i){
      tau_i = fe_sigma_Self[stress].value(i,point);
      complianceDot_tau_i = compliance_values[point] * tau_i;
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
  	sigma_j = fe_sigma_Self[stress].value(j,point);
  	doubledotProduct = complianceDot_tau_i * sigma_j;
  	tau_in_sigma_in(i,j) += factor * JxW_at_point * doubledotProduct;
      }
    }
  }
}



template <int dim>
void
StiffSigmaFromU
( dealii::FullMatrix<elas::real> & tau_in_U_in,
  const dealii::FEValuesBase<dim> & fe_tau_Self,
  const dealii::FEValuesBase<dim> & fe_U_Self,
  const elas::real factor)
{

  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(tau_in_U_in.m(),n_tau_dofs);
  AssertDimension(tau_in_U_in.n(),n_U_dofs);

  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_tau_Self.get_JxW_values();

  for(unsigned int point = 0;
      point < fe_tau_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_tau_dofs; ++i){
      const auto div_tau_i = fe_tau_Self[stress].divergence(i,point);      
      for(unsigned int j = 0; j < n_U_dofs; ++j){
	const auto U_j = fe_U_Self[displacements].value(j,point);
	tau_in_U_in(i,j) += factor * div_tau_i * U_j * JxW[point];
      }      
    }
  }  
}


template <int dim>
void
StiffSigmaFromU_NEW
( dealii::FullMatrix<elas::real> & tau_in_U_in,
  const dealii::FEValuesBase<dim> & fe_tau_Self,
  const dealii::FEValuesBase<dim> & fe_U_Self,
  const elas::real factor)
{

  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(tau_in_U_in.m(),n_tau_dofs);
  AssertDimension(tau_in_U_in.n(),n_U_dofs);

  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_tau_Self.get_JxW_values();
  dealii::Tensor<1,dim> div_tau_i, U_j;
  elas::real dotproduct;
  
  for(unsigned int point = 0;
      point < fe_tau_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_tau_dofs; ++i){
      div_tau_i = fe_tau_Self[stress].divergence(i,point);      
      for(unsigned int j = 0; j < n_U_dofs; ++j){
	U_j = fe_U_Self[displacements].value(j,point);
	dotproduct = div_tau_i * U_j;
	dotproduct *= JxW[point];
	dotproduct *= factor;
	tau_in_U_in(i,j) += dotproduct;
      }      
    }
  }  
}

template <int dim>
void
StiffUFromSigma
( dealii::FullMatrix<elas::real> & V_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_V_Self,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const elas::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;  

  AssertDimension(V_in_sigma_in.m(),n_V_dofs);
  AssertDimension(V_in_sigma_in.n(),n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );  

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_sigma_Self.get_JxW_values();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_V_dofs; ++i){
      const auto symgrad_V_i = fe_V_Self[displacements].symmetric_gradient(i,point);      
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
	const auto sigma_j = fe_sigma_Self[stress].value(j,point);
	V_in_sigma_in(i,j) += factor * ( symgrad_V_i * sigma_j )* JxW[point];
      }
    }
  }
}


template <int dim>
void
StiffUFromSigma_NEW
( dealii::FullMatrix<elas::real> & V_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_V_Self,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const elas::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;  

  AssertDimension(V_in_sigma_in.m(),n_V_dofs);
  AssertDimension(V_in_sigma_in.n(),n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );  

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_sigma_Self.get_JxW_values();
  dealii::SymmetricTensor<2,dim> symgrad_V_i, sigma_j;  
  elas::real doubledot;
  
  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_V_dofs; ++i){
      symgrad_V_i = fe_V_Self[displacements].symmetric_gradient(i,point);      
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
	sigma_j = fe_sigma_Self[stress].value(j,point);
	doubledot = symgrad_V_i * sigma_j;
	doubledot *= factor;
	doubledot *= JxW[point];
	V_in_sigma_in(i,j) += doubledot;
      }
    }
  }
}


template <int dim>
void
numericalFluxSigmaFromU
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 dealii::FullMatrix<double> & tau_in_U_out,
 dealii::FullMatrix<double> & tau_out_U_in,
 dealii::FullMatrix<double> & tau_out_U_out,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_tau_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig
 )
{

  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(n_tau_dofs, fe_tau_Neig.dofs_per_cell);
  AssertDimension(n_U_dofs, fe_U_Neig.dofs_per_cell );
  
  AssertDimension(tau_in_U_in.m(), n_tau_dofs);
  AssertDimension(tau_in_U_in.n(), n_U_dofs);
  AssertDimension(tau_in_U_out.m(), n_tau_dofs);
  AssertDimension(tau_in_U_out.n(), n_U_dofs);
  AssertDimension(tau_out_U_in.m(), n_tau_dofs);
  AssertDimension(tau_out_U_in.n(), n_U_dofs);
  AssertDimension(tau_out_U_out.m(), n_tau_dofs);
  AssertDimension(tau_out_U_out.n(), n_U_dofs);

  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_U_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_tau_Neig.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  //Some might argue that it would be better performance wise to combine the insider's and outsider's
  //perspective, and they probably would be correct.  However the authors judgement is that this is much
  //easier to understand.  

  {  // From insider's perspective
    
    const auto & JxW = fe_tau_Self.get_JxW_values();
    const auto & normals = fe_tau_Self.get_all_normal_vectors();
  
    for(unsigned int point = 0;
	point < fe_U_Self.n_quadrature_points;
	++point){
      const auto & normalVector = fe_U_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){	
	for(unsigned int i = 0; i< fe_tau_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
	    tau_in_U_out(i,j) +=
	      JxW[point]
	      * fe_tau_Self[stress].value(i,point)
	      * fe_U_Neig[displacements].value(j,point)
	      * normals[point];
	      }
	}
      } else{	
	for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    tau_in_U_in(i,j) +=
	      JxW[point]
	      * (fe_tau_Self[stress].value(i,point)
		 * fe_U_Self[displacements].value(j,point)) 
	      * normals[point];
	  }
	}
      }      
    }
  }

  { //From outsider's perspective 
    const auto & JxW = fe_tau_Neig.get_JxW_values();
    const auto & normals = fe_tau_Neig.get_all_normal_vectors();

    for(unsigned int point = 0; point < fe_tau_Neig.n_quadrature_points; ++point){
      const auto & normalVector = fe_tau_Neig.normal_vector(point);

      // for(int i = 0; i < dim; ++i){
      // 	PRINT( i );
      // 	PRINT( normalVector(i) );
      // }
      if(referenceDirection * normalVector > 0.0){
	
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    tau_out_U_in(i,j) += 
	      JxW[point]
	      * fe_tau_Neig[stress].value(i,point)
	      * fe_U_Self[displacements].value(j,point)
	      * normals[point];	      
	  }
	}
      } else {
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
	    tau_out_U_out(i,j) +=
	      JxW[point]
	      * fe_tau_Neig[stress].value(i,point)
	      * fe_U_Neig[displacements].value(j,point)
	      * normals[point];
	  }
	}
      }
    }
    
  }
}

template <int dim>
void
numericalFluxSigmaFromU_NEW
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 dealii::FullMatrix<double> & tau_in_U_out,
 dealii::FullMatrix<double> & tau_out_U_in,
 dealii::FullMatrix<double> & tau_out_U_out,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_tau_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig
 )
{

  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;  
  
  AssertDimension(n_tau_dofs, fe_tau_Neig.dofs_per_cell);
  AssertDimension(n_U_dofs, fe_U_Neig.dofs_per_cell );
  
  AssertDimension(tau_in_U_in.m(), n_tau_dofs);
  AssertDimension(tau_in_U_in.n(), n_U_dofs);
  AssertDimension(tau_in_U_out.m(), n_tau_dofs);
  AssertDimension(tau_in_U_out.n(), n_U_dofs);
  AssertDimension(tau_out_U_in.m(), n_tau_dofs);
  AssertDimension(tau_out_U_in.n(), n_U_dofs);
  AssertDimension(tau_out_U_out.m(), n_tau_dofs);
  AssertDimension(tau_out_U_out.n(), n_U_dofs);

  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_U_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_tau_Neig.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);
  dealii::SymmetricTensor<2,dim> sigma_in_i, sigma_out_i;
  dealii::Tensor<1,dim> U_in_j, U_out_j, sigma_times_normal;
  elas::real dotproduct;  

  //Some might argue that it would be better performance wise to combine the insider's and outsider's
  //perspective, and they probably would be correct.  However the authors judgement is that this is much
  //easier to understand.  

  {  // From insider's perspective
    
    const auto & JxW = fe_tau_Self.get_JxW_values();
    const auto & normals = fe_tau_Self.get_normal_vectors();
  
    for(unsigned int point = 0;
	point < fe_U_Self.n_quadrature_points;
	++point){
      const auto & normalVector = fe_U_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){
	for(unsigned int i = 0; i< fe_tau_Self.dofs_per_cell; ++i){
	  sigma_in_i = fe_tau_Self[stress].value(i,point);
	  sigma_times_normal = sigma_in_i * normalVector;
	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
	    U_out_j = fe_U_Neig[displacements].value(j,point);
	    dotproduct = sigma_times_normal * U_out_j;
	    dotproduct *= JxW[point];
	    
	    tau_in_U_out(i,j) += dotproduct;	    
	  }
	}
      } else{	
	for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
	  sigma_in_i = fe_tau_Self[stress].value(i,point);
	  sigma_times_normal = sigma_in_i * normals[point];
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    U_in_j = fe_U_Self[displacements].value(j,point);
	    dotproduct = sigma_times_normal * U_in_j;
	    dotproduct *= JxW[point];
	    tau_in_U_in(i,j) += dotproduct;
	  }
	}
      }      
    }
  }

  { //From outsider's perspective 
    const auto & JxW = fe_tau_Neig.get_JxW_values();
    const auto & normals = fe_tau_Neig.get_normal_vectors();

    for(unsigned int point = 0; point < fe_tau_Neig.n_quadrature_points; ++point){
      
      const auto & normalVector = fe_tau_Neig.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){		
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  sigma_out_i = fe_tau_Neig[stress].value(i,point);
	  sigma_times_normal = sigma_out_i * normalVector;
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    U_in_j = fe_U_Self[displacements].value(j,point);
	    dotproduct = sigma_times_normal * U_in_j;
	    dotproduct *= JxW[point];
	    tau_out_U_in(i,j) += dotproduct;     
	  }
	}
      } else {	
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  sigma_out_i = fe_tau_Neig[stress].value(i,point);
	  sigma_times_normal = sigma_out_i * normalVector;	  
	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
	    U_out_j = fe_U_Neig[displacements].value(j,point);
	    dotproduct = sigma_times_normal * U_out_j;
	    dotproduct *= JxW[point];
	    tau_out_U_out(i,j) += dotproduct;
	  }
	}
      }
    }
    
  }
}

template <int dim>
void
numericalFluxSigmaFromUBoundary
(const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self
 )
{
  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  AssertDimension(tau_in_U_in.m(), n_tau_dofs);
  AssertDimension(tau_in_U_in.n(), n_U_dofs);
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_tau_Self.get_JxW_values();
  const auto & normals = fe_tau_Self.get_all_normal_vectors();
  
  for(unsigned int point = 0;
      point < fe_U_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_U_Self.normal_vector(point);    
    Assert(referenceDirection * normalVector < 0.0, dealii::ExcInternalError() );

    for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	tau_in_U_in(i,j) +=
	  JxW[point]
	  * (fe_tau_Self[stress].value(i,point)
	     * fe_U_Self[displacements].value(j,point))
	  * normals[point];
      }
    }    
  }
}


template <int dim>
void
numericalFluxSigmaFromUBoundary_NEW
(const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self
 )
{
  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  AssertDimension(tau_in_U_in.m(), n_tau_dofs);
  AssertDimension(tau_in_U_in.n(), n_U_dofs);
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  dealii::SymmetricTensor<2,dim> sigma_in_i;
  dealii::Tensor<1,dim> U_in_j, sigma_times_normal;
  elas::real dotproduct;  

  const auto & JxW = fe_tau_Self.get_JxW_values();
  const auto & normals = fe_tau_Self.get_normal_vectors();  
  
  for(unsigned int point = 0;
      point < fe_U_Self.n_quadrature_points;
      ++point){
    
    const auto & normalVector = fe_U_Self.normal_vector(point);    
    Assert(referenceDirection * normalVector < 0.0, dealii::ExcInternalError() );
    
    for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
      sigma_in_i = fe_tau_Self[stress].value(i,point);
      sigma_times_normal = sigma_in_i * normalVector;
      for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	U_in_j = fe_U_Self[displacements].value(j,point);
	dotproduct = sigma_times_normal * U_in_j;
	dotproduct *= JxW[point];
	tau_in_U_in(i,j) +=
	  dotproduct;
      }
    }    
  }
}

template <int dim>
void
numericalFluxUFromSigma
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 dealii::FullMatrix<double> & V_in_sigma_out,
 dealii::FullMatrix<double> & V_out_sigma_in,
 dealii::FullMatrix<double> & V_out_sigma_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Neig
 )

{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  
  AssertDimension(n_V_dofs, fe_V_Neig.dofs_per_cell );  
  AssertDimension(n_sigma_dofs, fe_sigma_Neig.dofs_per_cell);

  
  AssertDimension(V_in_sigma_in.m(), n_V_dofs);
  AssertDimension(V_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(V_in_sigma_out.m(), n_V_dofs);
  AssertDimension(V_in_sigma_out.n(), n_sigma_dofs);
  AssertDimension(V_out_sigma_in.m(), n_V_dofs);
  AssertDimension(V_out_sigma_in.n(), n_sigma_dofs);
  AssertDimension(V_out_sigma_out.m(), n_V_dofs);
  AssertDimension(V_out_sigma_out.n(), n_sigma_dofs);

  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_V_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_sigma_Neig.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  {  // From insider's perspective
    
    const auto & JxW = fe_V_Self.get_JxW_values();
    const auto & normals = fe_V_Self.get_all_normal_vectors();
  
    for(unsigned int point = 0;
	point < fe_V_Self.n_quadrature_points;
	++point){
      const auto & normalVector = fe_sigma_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){
	for(unsigned int i = 0; i< fe_V_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	    V_in_sigma_in(i,j) += 
	      JxW[point]
	      * fe_V_Self[displacements].value(i,point)
	      * (fe_sigma_Self[stress].value(j,point)
		 * normals[point]);
	      }
	}
      } else{
	for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	    V_in_sigma_out(i,j) +=
	      JxW[point]
	      * fe_V_Self[displacements].value(i,point)
	      * (fe_sigma_Neig[stress].value(j,point)
		 * normals[point]);
	  }
	}

      }      
    }
  }

  { //From outsider's perspective 

    const auto & JxW = fe_V_Neig.get_JxW_values();
    const auto & normals = fe_V_Neig.get_all_normal_vectors();

    for(unsigned int point = 0; point < fe_V_Neig.n_quadrature_points; ++point){
      const auto & normalVector = fe_V_Neig.normal_vector(point);      
      if(referenceDirection * normalVector > 0){
	for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	    V_out_sigma_out(i,j) +=
	      JxW[point]
	      * fe_V_Neig[displacements].value(i,point)
	      * (fe_sigma_Neig[stress].value(j,point)
		 * normals[point]);
	  }
	}
      } else {
	for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	    V_out_sigma_in(i,j) +=
	      JxW[point]
	      * fe_V_Neig[displacements].value(i,point)
	      * (fe_sigma_Self[stress].value(j,point)
		 * normals[point]);
	  }
	}
      }
    }    
  }
}
template <int dim>
void
numericalFluxUFromSigma_NEW
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 dealii::FullMatrix<double> & V_in_sigma_out,
 dealii::FullMatrix<double> & V_out_sigma_in,
 dealii::FullMatrix<double> & V_out_sigma_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Neig
 )

{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  
  AssertDimension(n_V_dofs, fe_V_Neig.dofs_per_cell );  
  AssertDimension(n_sigma_dofs, fe_sigma_Neig.dofs_per_cell);

  
  AssertDimension(V_in_sigma_in.m(), n_V_dofs);
  AssertDimension(V_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(V_in_sigma_out.m(), n_V_dofs);
  AssertDimension(V_in_sigma_out.n(), n_sigma_dofs);
  AssertDimension(V_out_sigma_in.m(), n_V_dofs);
  AssertDimension(V_out_sigma_in.n(), n_sigma_dofs);
  AssertDimension(V_out_sigma_out.m(), n_V_dofs);
  AssertDimension(V_out_sigma_out.n(), n_sigma_dofs);

  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_V_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_sigma_Neig.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  dealii::Tensor<1,dim> V_in_i, V_out_i, tau_times_normal;
  dealii::SymmetricTensor<2,dim> tau_in_j, tau_out_j;
  elas::real dotproduct;

  {  // From insider's perspective
    
    const auto & JxW = fe_V_Self.get_JxW_values();
    const auto & normals = fe_V_Self.get_normal_vectors();
  
    for(unsigned int point = 0;
	point < fe_V_Self.n_quadrature_points;
	++point){
      const auto & normalVector = fe_sigma_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){
	//Perhaps annoyingly, we are going have the i loop (test function) be the inner most loop;
	//This is on purpose.
	for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	  tau_in_j = fe_sigma_Self[stress].value(j,point);
	  tau_times_normal = tau_in_j * normalVector;	  
	  for(unsigned int i = 0; i< fe_V_Self.dofs_per_cell; ++i){
	    V_in_i = fe_V_Self[displacements].value(i,point);
	    dotproduct = tau_times_normal * V_in_i;
	    dotproduct *= JxW[point];	    
	    V_in_sigma_in(i,j) += dotproduct;
	  }
	}
      } else{
	for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	  tau_out_j = fe_sigma_Neig[stress].value(j,point);
	  tau_times_normal = tau_out_j * normalVector;
	  for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
	    V_in_i = fe_V_Self[displacements].value(i,point);
	    dotproduct = tau_times_normal * V_in_i;
	    dotproduct *= JxW[point];
	  }
	}
      }      
    }
  }

  { //From outsider's perspective 

    const auto & JxW = fe_V_Neig.get_JxW_values();
    const auto & normals = fe_V_Neig.get_normal_vectors();

    for(unsigned int point = 0; point < fe_V_Neig.n_quadrature_points; ++point){
      const auto & normalVector = fe_V_Neig.normal_vector(point);      
      if(referenceDirection * normalVector > 0){
	for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	  tau_out_j = fe_sigma_Neig[stress].value(j,point);
	  tau_times_normal = tau_out_j * normalVector;
	  for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	    V_out_i = fe_V_Neig[displacements].value(i,point);
	    dotproduct = tau_times_normal * V_out_i;
	    dotproduct *= JxW[point];
	    V_out_sigma_out(i,j) +=  dotproduct;
	  }
	}
      } else {
	for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	  tau_in_j = fe_sigma_Self[stress].value(j,point);
	  tau_times_normal = tau_in_j * normalVector;
	  for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	    V_out_i = fe_V_Neig[displacements].value(i,point);
	    dotproduct = tau_times_normal * V_out_i;
	    dotproduct *= JxW[point];
	    
	    V_out_sigma_in(i,j) += dotproduct;
	    
	  }
	}
      }
    }    
  }
}

template <int dim>
void
numericalFluxUFromSigmaBoundary
(const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Self)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;

  AssertDimension(V_in_sigma_in.m(), n_V_dofs);
  AssertDimension(V_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  const auto & JxW = fe_V_Self.get_JxW_values();
  const auto & normals = fe_V_Self.get_all_normal_vectors();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_sigma_Self.normal_vector(point);
    Assert( referenceDirection * normalVector > 0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	V_in_sigma_in(i,j) +=
	  JxW[point]
	  * fe_V_Self[displacements].value(i,point)
	  * (fe_sigma_Self[stress].value(j,point)
	     * normals[point] );
      }
    }
  }  
}

template <int dim>
void
numericalFluxUFromSigmaBoundary_NEW
(const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Self)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;

  AssertDimension(V_in_sigma_in.m(), n_V_dofs);
  AssertDimension(V_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector displacements(0);

  dealii::Tensor<1,dim> V_in_i, sigma_times_normal;
  dealii::SymmetricTensor<2,dim> sigma_in_j;
  elas::real dotproduct;

  const auto & JxW = fe_V_Self.get_JxW_values();
  const auto & normals = fe_V_Self.get_normal_vectors();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_sigma_Self.normal_vector(point);
    Assert( referenceDirection * normalVector > 0, dealii::ExcInternalError() );
    for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
      sigma_in_j = fe_sigma_Self[stress].value(j,point);
      sigma_times_normal = sigma_in_j * normalVector;      
      
      for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
	V_in_i = fe_V_Self[displacements].value(i,point);
	dotproduct = sigma_times_normal * V_in_i;
	dotproduct *= JxW[point];
	
	V_in_sigma_in(i,j) += dotproduct;
      }
    }
  }  
}

template<int dim>
void
sigma_MinusRHS_FromDirPlus
(const dealii::Point<dim> referenceDirection,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 dealii::Vector<elas::real> & sigma_Vector)
{

  const elas::DirBC<dim> dirBC;

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);

  const auto & normals = fe_sigma_Self.get_all_normal_vectors();
  const auto & JxW = fe_sigma_Self.get_JxW_values();
  for(unsigned int point = 0; point < fe_sigma_Self.n_quadrature_points; ++point){
    Assert(referenceDirection * normals[point] > 0.0, dealii::ExcInternalError() );    
    for(unsigned int i = 0; i < fe_sigma_Self.dofs_per_cell; ++i){
        sigma_Vector(i) +=
    	JxW[point]
    	* (fe_sigma_Self[stress].value(i,point) * normals[point] )
    	* dirBC.value(fe_sigma_Self.quadrature_point(point));
    }
  }  

}

template <int dim>
void
U_MinusRHS_FromNueMinus
(const dealii::Point<dim> referenceDirection,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 dealii::Vector<elas::real> & U_Vector)
{
  const elas::NueBC<dim> nueBC;
  const dealii::FEValuesExtractors::Vector displacements(0);
  
  const auto & normals = fe_U_Self.get_all_normal_vectors();
  const auto & JxW = fe_U_Self.get_JxW_values();
  for(unsigned int point = 0; point < fe_U_Self.n_quadrature_points; ++point){
    Assert( referenceDirection * normals[point] < 0.0, dealii::ExcInternalError() );
    const auto temp1 = nueBC.value(fe_U_Self.quadrature_point(point));
    const auto temp2 = temp1 * normals[point];
    for(unsigned int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
  
      const auto temp3 = temp2 * fe_U_Self[displacements].value(i,point);
      
      U_Vector(i) += temp3 * JxW[point];

    }
  }
}

template<int dim>
void
ConstraintUDirMinus
(const dealii::Point<dim> referenceDirection,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 dealii::Vector<elas::real> & U_Vector)
{

  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  
  AssertDimension(U_Vector.size(), n_U_dofs);
  
  

  const elas::DirBC<dim> dirBC;
  const dealii::FEValuesExtractors::Vector displacements(0);
  const auto & JxW = fe_U_Self.get_JxW_values();
  const auto & normals = fe_U_Self.get_all_normal_vectors();
  for(unsigned int point = 0; point < fe_U_Self.n_quadrature_points; ++point){
    Assert(referenceDirection * normals[point] < 0.0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
      U_Vector(i) +=
	JxW[point] *
	(
	 fe_U_Self[displacements].value(i,point)
	 * dirBC.value(fe_U_Self.quadrature_point(point))
	 );
    }
  }
}

template <int dim>
void Constraint_sigma_NuePlus
(
 const dealii::Point<dim> & referenceDirection,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 dealii::Vector<elas::real> & U_Vector)
{
  const elas::NueBC<dim> nueBC;
  const auto & JxW = fe_U_Self.get_JxW_values();
  const auto & normals = fe_U_Self.get_all_normal_vectors();
  const dealii::FEValuesExtractors::Vector displacements(0);

  for(unsigned int point = 0; point < fe_U_Self.n_quadrature_points; ++point){
    Assert(referenceDirection * normals[point] > 0.0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
      U_Vector(i) +=
	JxW[point]
	* (
	   (nueBC.value(fe_U_Self.quadrature_point(point)) * normals[point])
	   * fe_U_Self[displacements].value(i,point) );
    }
  }
  
}

template<int dim>
void
BoundaryMassU
(const dealii::Point<dim> referenceDirection, 
 const dealii::FEValuesBase<dim> & fe_U_Self,
 dealii::FullMatrix<double> & U_in_U_in
 )
{

  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(U_in_U_in.m(),n_U_dofs);  
  AssertDimension(U_in_U_in.n(),n_U_dofs);
  
  //TODO::  We need some asserts here.
  
  const dealii::FEValuesExtractors::Vector displacements(0);


  const auto & JxW = fe_U_Self.get_JxW_values();
  const auto & normals = fe_U_Self.get_all_normal_vectors();

  for(unsigned int point = 0;
      point < fe_U_Self.n_quadrature_points;
      ++point){
    Assert(referenceDirection * normals[point] < 0.0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	U_in_U_in(i,j) +=
	  JxW[point] * fe_U_Self[displacements].value(i,point) * fe_U_Self[displacements].value(j,point);
	//std::cout << fe_U_Self[displacements].value(i,point) << std::endl;
	
      }
    }
  }
  
}

template<int dim>
void
BoundaryMassSigma
(const dealii::Point<dim> referenceDirection, 
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 dealii::FullMatrix<double> & tau_in_sigma_in)
{
  const unsigned int n_tau_dofs = fe_sigma_Self.dofs_per_cell;  
  //const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(tau_in_sigma_in.m(),n_tau_dofs);  
  AssertDimension(tau_in_sigma_in.n(),n_tau_dofs);
  
  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  
  const auto & JxW = fe_sigma_Self.get_JxW_values();
  const auto & normals = fe_sigma_Self.get_all_normal_vectors();

  for(unsigned int point = 0;
      point < fe_sigma_Self.n_quadrature_points;
      ++point){
    Assert(referenceDirection * normals[point] > 0, dealii::ExcInternalError() );
    
    for(unsigned int i = 0; i < fe_sigma_Self.dofs_per_cell; ++i){
      auto const temp2 = fe_sigma_Self[stress].value(i,point) * normals[point];
      for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	auto const temp = fe_sigma_Self[stress].value(j,point) * normals[point];
	
  	tau_in_sigma_in(i,j) +=
  	  JxW[point] * temp * temp2;	
      }
    }
  }
}

template<int dim>
void
SigmaJump (const dealii::FEValuesBase<dim> & fe_sigma_self,
	   const dealii::FEValuesBase<dim> & fe_sigma_neig,
	   const dealii::Vector<elas::real> & sigma_vector_self,
	   const dealii::Vector<elas::real> & sigma_vector_neig,
	   elas::real & jumpSquared)
{
  const unsigned int n_sigma_dofs = fe_sigma_self.dofs_per_cell;
  
  AssertDimension(fe_sigma_self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_sigma_neig.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(sigma_vector_self.size(),n_sigma_dofs);
  AssertDimension(sigma_vector_neig.size(),n_sigma_dofs);

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);

  const auto & JxW = fe_sigma_self.get_JxW_values();
  const auto & normals = fe_sigma_self.get_all_normal_vectors();  

  for(unsigned int point = 0;
      point < fe_sigma_self.n_quadrature_points;
      ++point){
    
    for(unsigned int i = 0; i < fe_sigma_self.dofs_per_cell; ++i){
      const auto fe_sigma_i =
	sigma_vector_self(i) * fe_sigma_self[stress].value(i,point);
      const auto & fe_sigma_i_dot_normal = fe_sigma_i * normals[point];
      
      for(unsigned int j = 0; j < fe_sigma_neig.dofs_per_cell; ++j){
	const auto & fe_sigma_j =
	  sigma_vector_neig(j) * fe_sigma_neig[stress].value(j,point);
	const auto & fe_sigma_j_dot_normal = fe_sigma_j * normals[point];

	const auto diff = fe_sigma_j_dot_normal - fe_sigma_i_dot_normal;
	
	const auto diff_dot_diff = diff * diff;
	jumpSquared += JxW[point] * diff_dot_diff;	
      }	
    }    
  }    
}

template<int dim>
void
CellResidualSigma
(const dealii::FEValuesBase<dim> & fe_sigma_self,
 const dealii::Vector<elas::real> & sigma_vector_self,
 elas::real & normSquaredOfCellResidual){

  const unsigned int n_sigma_dofs = fe_sigma_self.dofs_per_cell;

  AssertDimension(fe_sigma_self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(sigma_vector_self.size() ,n_sigma_dofs);

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const auto & JxW = fe_sigma_self.get_JxW_values();
  
  const elasRHS<dim> bodyValues;
  
  elas::real partial_sum = 0;

  for(unsigned int point = 0; point < fe_sigma_self.n_quadrature_points; ++point){
    const auto & point_in_space = fe_sigma_self.quadrature_point(point);
    const auto & bodyValueAtPoint = bodyValues.value(point_in_space);

    for(unsigned int i = 0; i < fe_sigma_self.dofs_per_cell; ++i){      
      const auto
	fe_sigma_i_div = sigma_vector_self(i) * fe_sigma_self[stress].divergence(i,point);
      
      const auto
	residual = fe_sigma_i_div - bodyValueAtPoint;
      const auto residual_dot_residual = residual * residual;

      partial_sum += residual_dot_residual * JxW[point];      
    }
    normSquaredOfCellResidual = partial_sum;
  }  
}

} //End namespace LDG
} //End namespace LocalIntegrators
} //End namespace elas




 

