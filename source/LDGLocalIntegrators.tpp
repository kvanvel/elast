#ifndef LDG_LOCAL_INTEGRATORS_T_
#define LDG_LOCAL_INTEGRATORS_T_

namespace stokes {
namespace LocalIntegrators {
namespace LDG {

template<int dim>
void
massSigma( dealii::FullMatrix<stokes::real> & tau_in_sigma_in,
	   const dealii::FEValuesBase<dim> & fe_sigma_Self,
	   const real factor)
{  
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;
  AssertDimension(tau_in_sigma_in.m(), n_sigma_dofs);
  AssertDimension(tau_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim ) /2 );

  
  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const auto & JxW = fe_sigma_Self.get_JxW_values();
  

  for(unsigned int point = 0;
      point < fe_sigma_Self.n_quadrature_points;  ++point){  
    for(unsigned int i = 0; i < n_sigma_dofs; ++i){      
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
	const dealii::SymmetricTensor<2,dim> tau_i = fe_sigma_Self[stress].value(i,point);      
  	const dealii::SymmetricTensor<2,dim> sigma_j = fe_sigma_Self[stress].value(j,point);
  	tau_in_sigma_in(i,j) += tau_i * JxW[point] * sigma_j * factor;
      }
    }
  }
}

  
template <int dim>
void
StiffSigmaFromU
( dealii::FullMatrix<stokes::real> & tau_in_U_in,
  const dealii::FEValuesBase<dim> & fe_tau_Self,
  const dealii::FEValuesBase<dim> & fe_U_Self,
  const stokes::real factor)
{

  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(tau_in_U_in.m(),n_tau_dofs);
  AssertDimension(tau_in_U_in.n(),n_U_dofs);

  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_tau_Self.get_JxW_values();

  for(unsigned int point = 0;
      point < fe_tau_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_tau_dofs; ++i){
      const auto div_tau_i = fe_tau_Self[stress].divergence(i,point);      
      for(unsigned int j = 0; j < n_U_dofs; ++j){
	const auto U_j = fe_U_Self[velocities].value(j,point);
	tau_in_U_in(i,j) += factor * div_tau_i * U_j * JxW[point];
      }      
    }
  }  
}

template <int dim>
void
StiffUFromSigma
( dealii::FullMatrix<stokes::real> & V_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_V_Self,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const stokes::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;  

  AssertDimension(V_in_sigma_in.m(),n_V_dofs);
  AssertDimension(V_in_sigma_in.n(),n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );  

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_sigma_Self.get_JxW_values();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    for(unsigned int i = 0; i < n_V_dofs; ++i){
      const auto symgrad_V_i = fe_V_Self[velocities].symmetric_gradient(i,point);      
      for(unsigned int j = 0; j < n_sigma_dofs; ++j){
	const auto sigma_j = fe_sigma_Self[stress].value(j,point);
	V_in_sigma_in(i,j) += factor * ( symgrad_V_i * sigma_j )* JxW[point];
      }
    }
  }
}

template <int dim>
void
StiffUFromP(dealii::FullMatrix<stokes::real> & V_in_P_in,
	    const dealii::FEValuesBase<dim> & fe_V_Self,
	    const dealii::FEValuesBase<dim> & fe_P_Self,
	    const stokes::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_P_dofs = fe_P_Self.dofs_per_cell;

  AssertDimension(V_in_P_in.m(), n_V_dofs);
  AssertDimension(V_in_P_in.n(),n_P_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_P_Self.get_fe().n_components(), 1);

  const dealii::FEValuesExtractors::Vector velocities(0);
  const dealii::FEValuesExtractors::Scalar pressure(0);

  const auto & JxW = fe_V_Self.get_JxW_values();
  for(unsigned int point = 0; point < fe_V_Self.n_quadrature_points; ++point){
    for(unsigned int i = 0; i < n_V_dofs; ++i){
      for(unsigned int j = 0; j < n_P_dofs; ++j){
	V_in_P_in(i,j)
	  += factor
	  * fe_V_Self[velocities].divergence(i,point)
	  * fe_P_Self[pressure].value(j,point)
	  * JxW[point];
      }
    }
  }
}

template <int dim>
void
StiffPFromU(dealii::FullMatrix<stokes::real> & P_in_U_in,
	    const dealii::FEValuesBase<dim> & fe_R_Self,
	    const dealii::FEValuesBase<dim> & fe_U_Self,
	    const stokes::real factor)
{
  const unsigned int n_R_dofs = fe_R_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  AssertDimension(P_in_U_in.m(),n_R_dofs);
  AssertDimension(P_in_U_in.n(),n_U_dofs);
  AssertDimension(fe_R_Self.get_fe().n_components(), 1);
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);

  const dealii::FEValuesExtractors::Scalar pressure(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_R_Self.get_JxW_values();
  for(unsigned int point = 0; point < fe_R_Self.n_quadrature_points; ++point){
    for(unsigned int i = 0; i < n_R_dofs; ++i){
      for(unsigned int j = 0; j < n_U_dofs; ++j){
	P_in_U_in(i,j)
	  += factor
	  * fe_R_Self[pressure].gradient(i,point)
	  * fe_U_Self[velocities].value(j,point)
	  * JxW[point];	  
      }
    }
  }
}

template <int dim>
void
numericalTraceSigmaFromU
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 dealii::FullMatrix<double> & tau_in_U_out,
 dealii::FullMatrix<double> & tau_out_U_in,
 dealii::FullMatrix<double> & tau_out_U_out,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_tau_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig,
 const stokes::real factor)
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
  const dealii::FEValuesExtractors::Vector velocities(0);

  //Some might argue that it would be better performance wise to combine the insider's and outsider's
  //perspective, and they probably would be correct. However, this is easier to understand.

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
	      factor
	      * JxW[point]
	      * fe_tau_Self[stress].value(i,point)
	      * fe_U_Neig[velocities].value(j,point)
	      * normals[point];
	      }
	}
      } else{	
	for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    tau_in_U_in(i,j) +=
	      factor
	      * JxW[point]
	      * fe_tau_Self[stress].value(i,point)
	      * fe_U_Self[velocities].value(j,point)
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
      if(referenceDirection * normalVector > 0.0){
	
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	    tau_out_U_in(i,j)
	      += factor
	      * JxW[point]
	      * fe_tau_Neig[stress].value(i,point)
	      * fe_U_Self[velocities].value(j,point)
	      * normals[point];	      
	  }
	}
      } else {
	for(unsigned int i = 0; i < fe_tau_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
	    tau_out_U_out(i,j)
	      += factor
	      * JxW[point]
	      * fe_tau_Neig[stress].value(i,point)
	      * fe_U_Neig[velocities].value(j,point)
	      * normals[point];
	  }
	}
      }
    }    
  }
}

template <int dim>
void
numericalTraceSigmaFromUBoundary
(const dealii::Tensor<1,dim> & referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const stokes::real factor)
{
  const unsigned int n_tau_dofs = fe_tau_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  AssertDimension(tau_in_U_in.m(), n_tau_dofs);
  AssertDimension(tau_in_U_in.n(), n_U_dofs);
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_tau_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_tau_Self.get_JxW_values();
  const auto & normals = fe_tau_Self.get_all_normal_vectors();
  
  for(unsigned int point = 0;
      point < fe_U_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_U_Self.normal_vector(point);    
    Assert(referenceDirection * normalVector < 0.0, dealii::ExcInternalError() );

    for(unsigned int i = 0; i < fe_tau_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	tau_in_U_in(i,j)
	  += factor
	  * JxW[point]
	  * fe_tau_Self[stress].value(i,point)
	  * fe_U_Self[velocities].value(j,point)
	  * normals[point];
      }
    }    
  }
}




template <int dim>
void
numericalTraceUFromSigma
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 dealii::FullMatrix<double> & V_in_sigma_out,
 dealii::FullMatrix<double> & V_out_sigma_in,
 dealii::FullMatrix<double> & V_out_sigma_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Neig,
 const stokes::real factor
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
	    V_in_sigma_in(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Self[displacements].value(i,point)
	      * fe_sigma_Self[stress].value(j,point)
	      * normals[point];
	      }
	}
      } else{
	for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Neig.dofs_per_cell; ++j){
	    V_in_sigma_out(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Self[displacements].value(i,point)
	      * fe_sigma_Neig[stress].value(j,point)
	      * normals[point];
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
	    V_out_sigma_out(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Neig[displacements].value(i,point)
	      * fe_sigma_Neig[stress].value(j,point)
	      * normals[point];
	  }
	}
      } else {
	for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	    V_out_sigma_in(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Neig[displacements].value(i,point)
	      * fe_sigma_Self[stress].value(j,point)
	      * normals[point];
	  }
	}
      }
    }    
  }
}


template <int dim>
void
numericalTraceUFromSigmaBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const stokes::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_sigma_dofs = fe_sigma_Self.dofs_per_cell;

  AssertDimension(V_in_sigma_in.m(), n_V_dofs);
  AssertDimension(V_in_sigma_in.n(), n_sigma_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_sigma_Self.get_fe().n_components(), (dim * dim + dim) / 2 );

  const dealii::FEValuesExtractors::SymmetricTensor<2> stress(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_V_Self.get_JxW_values();
  const auto & normals = fe_V_Self.get_all_normal_vectors();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_sigma_Self.normal_vector(point);
    Assert( referenceDirection * normalVector > 0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_sigma_Self.dofs_per_cell; ++j){
	V_in_sigma_in(i,j)
	  += factor
	  * JxW[point]
	  * fe_V_Self[velocities].value(i,point)
	  * fe_sigma_Self[stress].value(j,point)
	  * normals[point];
      }
    }
  }  
}

 

template< int dim>
void
numericalTraceUFromP
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_P_in,
 dealii::FullMatrix<double> & V_in_P_out,
 dealii::FullMatrix<double> & V_out_P_in,
 dealii::FullMatrix<double> & V_out_P_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_P_Self,
 const dealii::FEValuesBase<dim> & fe_P_Neig,
 const stokes::real factor)
{

  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_P_dofs = fe_P_Self.dofs_per_cell;
  
  AssertDimension(n_V_dofs, fe_V_Neig.dofs_per_cell );  
  AssertDimension(n_P_dofs, fe_P_Neig.dofs_per_cell);
  
  AssertDimension(V_in_P_in.m(), n_V_dofs);
  AssertDimension(V_in_P_in.n(), n_P_dofs);
  AssertDimension(V_in_P_out.m(), n_V_dofs);
  AssertDimension(V_in_P_out.n(), n_P_dofs);
  AssertDimension(V_out_P_in.m(), n_V_dofs);
  AssertDimension(V_out_P_in.n(), n_P_dofs);
  AssertDimension(V_out_P_out.m(), n_V_dofs);
  AssertDimension(V_out_P_out.n(), n_P_dofs);

  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_V_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_P_Self.get_fe().n_components(), 1);
  AssertDimension(fe_P_Neig.get_fe().n_components(), 1);

  const dealii::FEValuesExtractors::Vector velocities(0);
  const dealii::FEValuesExtractors::Scalar pressure(0);

  //insider
  {
    const auto & JxW =fe_V_Self.get_JxW_values();
    const auto & normals = fe_V_Self.get_all_normal_vectors();

    for(unsigned int point = 0;
	point < fe_V_Self.n_quadrature_points;
	++point){
      const auto & normalVector = fe_P_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){
	for(unsigned int i = 0; i< fe_V_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_P_Neig.dofs_per_cell; ++j){
	    V_in_P_in(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Self[velocities].value(i,point)
	      * fe_P_Self[pressure].value(j,point)
	      * normals[point];
	      }
	}
      } else{
	for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_P_Neig.dofs_per_cell; ++j){
	    V_in_P_out(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Self[velocities].value(i,point)
	      * fe_P_Neig[pressure].value(j,point)
	      * normals[point];
	  }
	}	
      }      
    }
  }

  //outside
  {
    const auto & JxW = fe_V_Neig.get_JxW_values();
    const auto & normals = fe_V_Neig.get_all_normal_vectors();

    for(unsigned int point = 0; point < fe_V_Neig.n_quadrature_points; ++point){
      const auto & normalVector = fe_V_Neig.normal_vector(point);      
      if(referenceDirection * normalVector > 0){
	for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_P_Neig.dofs_per_cell; ++j){
	    V_out_P_out(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Neig[velocities].value(i,point)
	      * fe_P_Neig[pressure].value(j,point)
	      * normals[point];
	  }
	}
      } else {
	for(unsigned int i = 0; i < fe_V_Neig.dofs_per_cell; ++i){
	  for(unsigned int j = 0; j < fe_P_Self.dofs_per_cell; ++j){
	    V_out_P_in(i,j)
	      += factor
	      * JxW[point]
	      * fe_V_Neig[velocities].value(i,point)
	      * fe_P_Self[pressure].value(j,point)
	      * normals[point];
	  }
	}
      }
    }    
  }
}

template <int dim>
void
numericalTraceUFromPBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_P_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_P_Self,
 const stokes::real factor)
{
  const unsigned int n_V_dofs = fe_V_Self.dofs_per_cell;
  const unsigned int n_P_dofs = fe_P_Self.dofs_per_cell;

  AssertDimension(V_in_P_in.m(), n_V_dofs);
  AssertDimension(V_in_P_in.n(), n_P_dofs);
  AssertDimension(fe_V_Self.get_fe().n_components(), dim);
  AssertDimension(fe_P_Self.get_fe().n_components(), 1);

  const dealii::FEValuesExtractors::Scalar pressure(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_V_Self.get_JxW_values();
  const auto & normals = fe_V_Self.get_all_normal_vectors();

  for(unsigned int point = 0;
      point < fe_V_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_P_Self.normal_vector(point);
    Assert( referenceDirection * normalVector > 0, dealii::ExcInternalError() );
    for(unsigned int i = 0; i < fe_V_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_P_Self.dofs_per_cell; ++j){
	V_in_P_in(i,j)
	  += factor
	  * JxW[point]
	  * fe_V_Self[velocities].value(i,point)
	  * fe_P_Self[pressure].value(j,point)
	  * normals[point];
      }
    }
  }  
}

  

template <int dim>
void
numericalTracePFromU
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & R_in_U_in,
 dealii::FullMatrix<double> & R_in_U_out,
 dealii::FullMatrix<double> & R_out_U_in,
 dealii::FullMatrix<double> & R_out_U_out,
 const dealii::FEValuesBase<dim> & fe_R_Self,
 const dealii::FEValuesBase<dim> & fe_R_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig,
 const stokes::real factor)
{

  
  const unsigned int n_R_dofs = fe_R_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(n_R_dofs, fe_R_Neig.dofs_per_cell);
  AssertDimension(n_U_dofs, fe_U_Neig.dofs_per_cell );
  
  AssertDimension(R_in_U_in.m(), n_R_dofs);
  AssertDimension(R_in_U_in.n(), n_U_dofs);
  AssertDimension(R_in_U_out.m(), n_R_dofs);
  AssertDimension(R_in_U_out.n(), n_U_dofs);
  AssertDimension(R_out_U_in.m(), n_R_dofs);
  AssertDimension(R_out_U_in.n(), n_U_dofs);
  AssertDimension(R_out_U_out.m(), n_R_dofs);
  AssertDimension(R_out_U_out.n(), n_U_dofs);

  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_U_Neig.get_fe().n_components(), dim);
  AssertDimension(fe_R_Self.get_fe().n_components(), 1);
  AssertDimension(fe_R_Neig.get_fe().n_components(), 1);

  const dealii::FEValuesExtractors::Scalar pressure(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  //assert(false);

  //Some might argue that it would be better performance wise to combine the insider's and outsider's
  //perspective, and they probably would be correct. However, this is easier to understand.

  {  // From insider's perspective
    
    const auto & JxW = fe_R_Self.get_JxW_values();
    const auto & normals = fe_R_Self.get_all_normal_vectors();
  
    for(unsigned int point = 0;
  	point < fe_U_Self.n_quadrature_points;
  	++point){
      const auto & normalVector = fe_U_Self.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){	
  	for(unsigned int i = 0; i< fe_R_Self.dofs_per_cell; ++i){
  	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
  	    R_in_U_out(i,j)
	      += factor
  	      * JxW[point]
  	      * fe_R_Self[pressure].value(i,point)
  	      * fe_U_Neig[velocities].value(j,point)
  	      * normals[point];
	  }
  	}
      } else{	
  	for(unsigned int i = 0; i < fe_R_Self.dofs_per_cell; ++i){
  	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
  	    R_in_U_in(i,j)
	      += factor
  	      * JxW[point]
  	      * fe_R_Self[pressure].value(i,point)
	      * fe_U_Self[velocities].value(j,point)
	      * normals[point];
	  }
	}
      }      
    }
  }
  

  { //From outsider's perspective 
    const auto & JxW = fe_R_Neig.get_JxW_values();
    const auto & normals = fe_R_Neig.get_all_normal_vectors();

    for(unsigned int point = 0; point < fe_R_Neig.n_quadrature_points; ++point){
      const auto & normalVector = fe_R_Neig.normal_vector(point);
      if(referenceDirection * normalVector > 0.0){
	
  	for(unsigned int i = 0; i < fe_R_Neig.dofs_per_cell; ++i){
  	  for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
  	    R_out_U_in(i,j)
	      += factor 
  	      * JxW[point]
  	      * fe_R_Neig[pressure].value(i,point)
  	      * fe_U_Self[velocities].value(j,point)
  	      * normals[point];	      
  	  }
  	}
      } else {
  	for(unsigned int i = 0; i < fe_R_Neig.dofs_per_cell; ++i){
  	  for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
  	    R_out_U_out(i,j)
	      += factor
	      * JxW[point]
  	      * fe_R_Neig[pressure].value(i,point)
  	      * fe_U_Neig[velocities].value(j,point)
  	      * normals[point];
  	  }
  	}
      }
    }    
  }

}


template <int dim>
void
numericalTracePFromUBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & R_in_U_in,
 const dealii::FEValuesBase<dim> & fe_R_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const stokes::real factor)
{
  const unsigned int n_R_dofs = fe_R_Self.dofs_per_cell;
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;
  
  AssertDimension(R_in_U_in.m(), n_R_dofs);
  AssertDimension(R_in_U_in.n(), n_U_dofs);
  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_R_Self.get_fe().n_components(), 1);

  const dealii::FEValuesExtractors::Scalar pressure(0);
  const dealii::FEValuesExtractors::Vector velocities(0);

  const auto & JxW = fe_R_Self.get_JxW_values();
  const auto & normals = fe_R_Self.get_all_normal_vectors();
  
  for(unsigned int point = 0;
      point < fe_U_Self.n_quadrature_points;
      ++point){
    const auto & normalVector = fe_U_Self.normal_vector(point);    
    Assert(referenceDirection * normalVector < 0.0, dealii::ExcInternalError() );

    for(unsigned int i = 0; i < fe_R_Self.dofs_per_cell; ++i){
      for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
	R_in_U_in(i,j)
	  += factor
	  * JxW[point]
	  * fe_R_Self[pressure].value(i,point)
	  * fe_U_Self[velocities].value(j,point)
	  * normals[point];
      }
    }    
  }  
}

template <int dim>
void
AdvectionBody(const dealii::FEValuesBase<dim> & fe_U_Self,
	      const dealii::Vector<stokes::real> & U_vector_Self,
	      dealii::Vector<stokes::real> & vector_Out)
{
  const unsigned int n_U_dofs = fe_U_Self.dofs_per_cell;

  AssertDimension(fe_U_Self.get_fe().n_componenent(), dim);
  AssertDimension(U_vector_Self.size(), n_U_dofs);
  AssertDimension(vector_Out.size(), n_U_dofs);

  const dealii::FEValuesExtractors::Vector velocities(0);
  const auto & JxW = fe_U_Self.get_JxW_values;
  for(unsigned int point = 0; point < fe_U_Self.n_quadrature_points; ++point){
    dealii::Tensor<1,dim> valueAtPoint;
    for(int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
      valueAtPoint += fe_U_Self[velocities].value(j,point) * U_vector_Self[j];
    }
    for(unsigned int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
      vector_Out[i]
	+= JxW[point]
	* fe_U_Self[velocities].gradient(i,point)
	* valueAtPoint
	* valueAtPoint;
    }
  }
}  


template <int dim>
void
AdvectionNumericalTrace(const dealii::FEValuesBase<dim> & fe_U_Self,
			const dealii::FEValuesBase<dim> & fe_U_Neig,
			const dealii::Vector<stokes::real> & U_vector_Self,
			const dealii::Vector<stokes::real> & U_vector_Neig,
			dealii::Vector<stokes::real> & vector_output)
{
  const unsigned int n_U_dofs_Self = fe_U_Self.dofs_per_cell;
  const unsigned int n_U_dofs_Neig = fe_U_Neig.dofs_per_cell;

  AssertDimension(fe_U_Self.get_fe().n_components(), dim);
  AssertDimension(fe_U_Neig.get_fe().n_components(), dim);
  AssertDimension(U_vector_Self.size(), n_U_dofs_Self);
  AssertDimension(U_vector_Neig.size(), n_U_dofs_Neig);

  AssertDimension(vector_output.size(), n_U_dofs_Self);

  const dealii::FEValuesExtractors::Vector velocities(0);
  const auto & JxW = fe_U_Self.get_JxW_values();
  const auto & normals = fe_U_Self.get_all_normal_vectors();
  for(unsigned int point = 0; point < fe_U_Self.n_quadrature_points; ++point){
    const auto & normalVector = normals[point];    

    dealii::Tensor<1,dim> valueAtPoint_Inside;
    for(unsigned int j = 0; j < fe_U_Self.dofs_per_cell; ++j){
      valueAtPoint_Inside += fe_U_Self[velocities].value(j,point)*U_vector_Self[j];
    }

    dealii::Tensor<1,dim> valueAtPoint_Outside;
    for(unsigned int j = 0; j < fe_U_Neig.dofs_per_cell; ++j){
      valueAtPoint_Outside += fe_U_Neig[velocities].value(j,point)*U_vector_Neig[j];
    }

    const dealii::Tensor<1,dim> average = (valueAtPoint_Inside + valueAtPoint_Outside)/2;

    dealii::Tensor<1,dim> valueAtPoint_upwind;
    const stokes::real averageDotNormal = average * normalVector;

    //Not tested;
    if(averageDotNormal > stokes::real(0)){
      valueAtPoint_upwind = valueAtPoint_Inside;
    } else {
      valueAtPoint_upwind = valueAtPoint_Outside;
    }

    for(int i = 0; i < fe_U_Self.dofs_per_cell; ++i){
      vector_output[i]
	+= JxW[point]
	* average
	* normalVector	
	* fe_U_Self[velocities].value(i,point)
	* valueAtPoint_upwind;      
    }

    
  }
		  
  
}

  


  





} //end namespace
} //end namespace
} //end namespace

#endif
