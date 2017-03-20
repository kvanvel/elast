#ifndef LDG_INTEGRATOR_T_
#define LDG_INTEGRATOR_T_

#include "LDGLocalIntegrators.hpp"
#include <deal.II/lac/lapack_full_matrix.h>

namespace stokes {
namespace LDGIntegrator {

template <int dim>
LDGIntegrator<dim>::LDGIntegrator
(
 const dealii::Tensor<1,dim> & referenceDirection_In,
 const dealii::Quadrature<dim-1> & face_quadrature_In,
 const dealii::UpdateFlags & face_update_flags_In,
 const dealii::MappingQ<dim,dim> & mapping_In,
 const std::map<dealii::types::boundary_id, stokes::myBoundaryID > & BoundaryIDMap_In)
  : dealii::MeshWorker::LocalIntegrator<dim>(true,true,true)
  , referenceDirection(referenceDirection_In)
  , face_quadrature(& face_quadrature_In)
  , face_update_flags( face_update_flags_In)
  , mapping(mapping_In)
  , BoundaryIDMap(BoundaryIDMap_In)
{}

template <int dim>
void
LDGIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> & dinfo,
			 dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const stokes::real one = stokes::real(1.0);
  const stokes::real minusOne = stokes::real(-1.0);
  
  stokes::LocalIntegrators::LDG::massSigma(dinfo.matrix(0,false).matrix,
					   info.fe_values(0),
					   one);

  {
    dealii::LAPACKFullMatrix<stokes::real> tempLAPACKMatrix;
    tempLAPACKMatrix.copy_from(dinfo.matrix(0,false).matrix);
    tempLAPACKMatrix.invert();
    dinfo.matrix(1,false).matrix = tempLAPACKMatrix;
    dinfo.matrix(1,false).matrix.symmetrize();
  }

  stokes
    ::LocalIntegrators
    ::LDG
    ::StiffSigmaFromU(dinfo.matrix(2,false).matrix,
		      info.fe_values(0),
		      info.fe_values(1),
		      one);

  stokes
    ::LocalIntegrators
    ::LDG
    ::StiffUFromSigma(dinfo.matrix(3,false).matrix,
		      info.fe_values(1),
		      info.fe_values(0),
		      minusOne);
    
  stokes
    ::LocalIntegrators
    ::LDG
    ::StiffUFromP(dinfo.matrix(4,false).matrix,
  		  info.fe_values(1),
  		  info.fe_values(2),
  		  one);

  stokes
    ::LocalIntegrators
    ::LDG
    ::StiffPFromU(dinfo.matrix(5,false).matrix,
  		  info.fe_values(2),
  		  info.fe_values(1),
  		  minusOne);
  
  
}

template <int dim>
void
LDGIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
			 dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
			 dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
			 dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const
{

  const stokes::real one = stokes::real(1.0);
  const stokes::real minusOne = stokes::real(-1.0);  
  
  stokes::LocalIntegrators::LDG::numericalTraceSigmaFromU
    (referenceDirection,
     dinfoSELF.matrix(2,false),
     dinfoSELF.matrix(2,true),
     dinfoNEIG.matrix(2,true),
     dinfoNEIG.matrix(2,false),
     infoSELF.fe_values(0),
     infoNEIG.fe_values(0),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1),
     minusOne
     );

  stokes::LocalIntegrators::LDG::numericalTraceUFromSigma
    (
     referenceDirection,
     dinfoSELF.matrix(3,false),
     dinfoSELF.matrix(3,true),
     dinfoNEIG.matrix(3,true),
     dinfoNEIG.matrix(3,false),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1),
     infoSELF.fe_values(0),
     infoNEIG.fe_values(0),
     one
     );

  stokes::LocalIntegrators::LDG::numericalTraceUFromP
    (
     referenceDirection,
     dinfoSELF.matrix(4,false),
     dinfoSELF.matrix(4,true),
     dinfoNEIG.matrix(4,true),
     dinfoNEIG.matrix(4,false),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1),
     infoSELF.fe_values(2),
     infoNEIG.fe_values(2),
     minusOne
     );

  stokes::LocalIntegrators::LDG::numericalTracePFromU
    (referenceDirection,
     dinfoSELF.matrix(5,false),
     dinfoSELF.matrix(5,true),
     dinfoNEIG.matrix(5,true),
     dinfoNEIG.matrix(5,false),
     infoSELF.fe_values(2),
     infoNEIG.fe_values(2),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1),
     one);

  
  

}

template<int dim>
void
LDGIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
			     dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
   const stokes::real one = stokes::real(1.0);
   const stokes::real minusOne = stokes::real(-1.0);
  

   switch (BoundaryIDMap.at(dinfo.face->boundary_id() ) ){
   case stokes::myBoundaryID::dir_Plus :
     stokes::LocalIntegrators::LDG::numericalTraceUFromSigmaBoundary
       (referenceDirection,
	dinfo.matrix(3,false),
	info.fe_values(1),
	info.fe_values(0),
	one);
     stokes::LocalIntegrators::LDG::numericalTraceUFromPBoundary
       (referenceDirection,
	dinfo.matrix(4,false),
	info.fe_values(1),
	info.fe_values(2),
	minusOne
	);
     
     break;
   
   case stokes::myBoundaryID::dir_Minus :
     stokes::LocalIntegrators::LDG::numericalTraceSigmaFromUBoundary
       (referenceDirection,
	dinfo.matrix(2,false),
	info.fe_values(0),
	info.fe_values(1),
	minusOne);
     stokes::LocalIntegrators::LDG::numericalTracePFromUBoundary
       (referenceDirection,
	dinfo.matrix(5,false),
	info.fe_values(2),
	info.fe_values(1),
	one);
    break;
    
   case stokes::myBoundaryID::nue_Plus :
     stokes::LocalIntegrators::LDG::numericalTraceUFromSigmaBoundary
       (referenceDirection,
	dinfo.matrix(3, false),
	info.fe_values(1),
	info.fe_values(0),
	one);
     stokes::LocalIntegrators::LDG::numericalTraceUFromPBoundary
       (referenceDirection,
	dinfo.matrix(4,false),
	info.fe_values(1),
	info.fe_values(2),
	minusOne
	);

     break;
   case stokes::myBoundaryID::nue_Minus :
     stokes::LocalIntegrators::LDG::numericalTraceSigmaFromUBoundary
       (referenceDirection,
	dinfo.matrix(2,false),
	info.fe_values(0),
	info.fe_values(1),
	one);

     stokes::LocalIntegrators::LDG::numericalTracePFromUBoundary
      (referenceDirection,
       dinfo.matrix(5,false),
       info.fe_values(2),
       info.fe_values(1),
       one);

     
     break;
   default :
     break;
   }

}




  

    
  
} //end namespace  
} //end namespace

#endif
