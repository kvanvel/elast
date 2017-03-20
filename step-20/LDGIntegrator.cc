#include "LDGIntegrator.hpp"
#include "LDG.hpp"

namespace elas{
namespace LDGIntegrator{

template <int dim>
LDGIntegrator<dim>::LDGIntegrator
(
 dealii::Tensor<1,dim>  referenceDirection_In, //Why not const reference?
 const dealii::Quadrature<dim-1> & face_quadrature_In,
 const dealii::UpdateFlags & face_update_flags_In,
 const dealii::MappingQ<dim,dim> & mapping_In,
 std::map<dealii::types::boundary_id, elas::myBoundaryID> const * const BoundaryIDMap_In)
  :
  dealii::MeshWorker::LocalIntegrator<dim>(true,true,true),
  referenceDirection(referenceDirection_In),
  face_quadrature(& face_quadrature_In),
  face_update_flags( face_update_flags_In),
  mapping(mapping_In),
  BoundaryIDMap(BoundaryIDMap_In)
{}

template<int dim>
void
LDGIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> & dinfo,
			 dealii::MeshWorker::IntegrationInfo<dim> &info ) const
{
  const elas::real one =  1.0;
  const elas::real minusOne = -1.0;

  //mass Sigma
  
  elas::LocalIntegrators::LDG::massSigma(dinfo.matrix(0,false).matrix,
					 info.fe_values(0),
					 one);

    
  //Invert the local mass matrix and insert it into the global inverse mass matrix  
  
  {
    //dinfo.matrix(0,false).matrix.symmetrize(); //Just to be safe.
  dealii::LAPACKFullMatrix<elas::real> tempLAPACKMatrix;

  tempLAPACKMatrix.copy_from(dinfo.matrix(0,false).matrix);
  tempLAPACKMatrix.invert();
  dinfo.matrix(5,false).matrix = tempLAPACKMatrix;
  dinfo.matrix(5,false).matrix.symmetrize(); //We are being really safe;
  
  }
  

  elas::LocalIntegrators::LDG::StiffSigmaFromU(dinfo.matrix(1,false).matrix,
						   info.fe_values(0),
						   info.fe_values(1),
						   -1.0);

  elas::LocalIntegrators::LDG::StiffUFromSigma(dinfo.matrix(2,false).matrix,
						   info.fe_values(1),
						   info.fe_values(0),
						   -1.0);

  for(unsigned int face_no = 0;
      face_no < dealii::GeometryInfo<dim>::faces_per_cell;
      ++face_no){
    typename dealii::Triangulation<dim>::face_iterator
      face_sys = dinfo.cell->face(face_no);
    if(face_sys->at_boundary() ){

      dealii::FEFaceValues<dim>
  	fe_face_U(mapping,
  		  info.finite_element().base_element(1),
  		  *face_quadrature,
  		  dealii::update_values
  		  | dealii::update_quadrature_points
  		  | dealii::update_JxW_values
  		  | dealii::update_normal_vectors);
      fe_face_U.reinit(dinfo.cell, face_no);

      dealii::FEFaceValues<dim>
  	fe_face_sigma(mapping,
  		      info.finite_element().base_element(0),
  		      *face_quadrature,
  		      dealii::update_values
  		      | dealii::update_quadrature_points
  		      | dealii::update_JxW_values
  		      | dealii::update_normal_vectors);
      fe_face_sigma.reinit(dinfo.cell, face_no);

      switch (BoundaryIDMap->at(face_sys->boundary_id() ) ){
      case elas::myBoundaryID::dir_Plus:
	
  	break;
      case elas::myBoundaryID::dir_Minus:
  	elas::LocalIntegrators::LDG::BoundaryMassU(referenceDirection,
  						   fe_face_U,
  						   dinfo.matrix(3,false) );

  	std::cout << "On dir_minus" << std::endl;
  	elas::LocalIntegrators::LDG::ConstraintUDirMinus(referenceDirection,
  							 fe_face_U,
  							 dinfo.vector(1).block(1) );
							 
							 
	
  	break;
      case elas::myBoundaryID::nue_Plus:
  	std::cout << "On Neu_plus" << std::endl;
  	elas::LocalIntegrators::LDG::BoundaryMassSigma(referenceDirection,
  						       fe_face_sigma,
  						       dinfo.matrix(4,false) );

	

  	elas::LocalIntegrators::LDG::Constraint_sigma_NuePlus(referenceDirection,
							      fe_face_sigma,
							      dinfo.vector(1).block(0) );
						       
  	// elas::LocalIntegrators::LDG::BoundaryMassU(referenceDirection,
  						   // 					   dinfo.matrix(4,false), 
  	// 					   fe_face_U);
  	break;
      case elas::myBoundaryID::nue_Minus:
  	break;
      default:
  	Assert(false, dealii::ExcInternalError() );
  	break;      
      }
      
    }    
  }

  
  
  { // Work for Boundary Mass U
    dealii::LAPACKFullMatrix<elas::real>
      tempLAPACKMatrix;
    tempLAPACKMatrix.copy_from( dinfo.matrix(3,false).matrix);
    tempLAPACKMatrix.compute_svd();
    dinfo.matrix(6,false).matrix = *(tempLAPACKMatrix.svd_u);
    dinfo.matrix(11,false).matrix = *(tempLAPACKMatrix.svd_vt);
    dealii::Vector<elas::real> vectorOfSingularValues(tempLAPACKMatrix.m() );

    for( unsigned int i = 0; i < tempLAPACKMatrix.m(); ++i){
      vectorOfSingularValues(i) = tempLAPACKMatrix.singular_value(i);
    }

    //Apply scaling, we have to do this because of problems when the mesh is very finely refined
    //We can't just assume that singular values on the order of 1e-13 are supposed to be zero.

    const elas::real zero = 0.0;
    const elas::real one = 1.0;

    elas::real scaling = vectorOfSingularValues.linfty_norm();
    if(scaling > zero){
      vectorOfSingularValues /= scaling;
    }
    else {
      scaling = one;
    }

    for(unsigned int i = 0; i < dinfo.matrix(7,false).matrix.m(); ++i){
      elas::real valueToInsert = zero;
      if(fabs(vectorOfSingularValues(i) ) > 1e-12){
    	valueToInsert = scaling *  vectorOfSingularValues(i);	
      }
      dinfo.matrix(7, false).matrix(i,i) = valueToInsert;
    }
  }

  //assert(false);
  
  { //Work for Boundary MassQ
    dealii::LAPACKFullMatrix<elas::real>
      tempLAPACKMatrix;
    tempLAPACKMatrix.copy_from( dinfo.matrix(4,false).matrix);
    tempLAPACKMatrix.compute_svd();
    dinfo.matrix(8,false).matrix = *(tempLAPACKMatrix.svd_u);
    dinfo.matrix(10,false).matrix = *(tempLAPACKMatrix.svd_vt);
    
    dealii::Vector<elas::real> vectorOfSingularValues(tempLAPACKMatrix.m() );

     for( unsigned int i = 0; i < tempLAPACKMatrix.m(); ++i){
       vectorOfSingularValues(i) = tempLAPACKMatrix.singular_value(i);
     }

    //Apply scaling, we have to do this because of problems when the mesh is very finely refined
    //We can't just assume that singular values on the order of 1e-13 are supposed to be zero.

     const elas::real zero = 0.0;
     const elas::real one = 1.0;

    elas::real scaling = vectorOfSingularValues.linfty_norm();
    if(scaling > zero){
      vectorOfSingularValues /= scaling;
    }
    else {
      scaling = one;
    }

    for(unsigned int i = 0; i < dinfo.matrix(9,false).matrix.m(); ++i){
      elas::real valueToInsert = zero;
      if(fabs(vectorOfSingularValues(i) ) > 1e-12){
    	valueToInsert = scaling *  vectorOfSingularValues(i);	
      }
      dinfo.matrix(9, false).matrix(i,i) = valueToInsert;
    }    
  }
  
}

template <int dim>
void
LDGIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
			 dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
			 dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
			 dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const {

  


  elas::LocalIntegrators::LDG::numericalFluxSigmaFromU
    (referenceDirection,
     dinfoSELF.matrix(1,false),
     dinfoSELF.matrix(1,true),
     dinfoNEIG.matrix(1,true),
     dinfoNEIG.matrix(1,false),
     infoSELF.fe_values(0),
     infoNEIG.fe_values(0),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1)     
     );

  elas::LocalIntegrators::LDG::numericalFluxUFromSigma
    (
     referenceDirection,
     dinfoSELF.matrix(2,false),
     dinfoSELF.matrix(2,true),
     dinfoNEIG.matrix(2,true),
     dinfoNEIG.matrix(2,false),
     infoSELF.fe_values(1),
     infoNEIG.fe_values(1),
     infoSELF.fe_values(0),
     infoNEIG.fe_values(0)
     );
}

template <int dim>
void
LDGIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
			     dealii::MeshWorker::IntegrationInfo<dim> & info) const
{
  switch (BoundaryIDMap->at(dinfo.face->boundary_id() )){
  case elas::myBoundaryID::dir_Plus:
    elas::LocalIntegrators::LDG::numericalFluxUFromSigmaBoundary
      (referenceDirection,
       dinfo.matrix(2,false),
       info.fe_values(1),
       info.fe_values(0) );

    elas::LocalIntegrators::LDG::sigma_MinusRHS_FromDirPlus
      (referenceDirection,
       info.fe_values(0),
       dinfo.vector(0).block(0) );
      
    break;
  case elas::myBoundaryID::dir_Minus:
    elas::LocalIntegrators::LDG::numericalFluxSigmaFromUBoundary
      (referenceDirection,
       dinfo.matrix(1,false),
       info.fe_values(0),
       info.fe_values(1)
       );

    // elas::LocalIntegrators::LDG::ConstraintUDirMinus
    //   (
    //    referenceDirection,
    //    info.fe_values(1),
    //    dinfo.vector(0).block(1) );  //TODO:  This is wrong
						     
    
    break;
  case elas::myBoundaryID::nue_Plus:
    elas::LocalIntegrators::LDG::numericalFluxUFromSigmaBoundary
      (referenceDirection,
       dinfo.matrix(2,false),
       info.fe_values(1),
       info.fe_values(0) );

    // elas::LocalIntegrators::LDG::Constraint_sigma_NuePlus
    //   (referenceDirection,
    //    info.fe_values(1),
    //    dinfo.vector(1).block(1) ); //TODO:  This is wrong
      

      
    break;
  case elas::myBoundaryID::nue_Minus:
    elas::LocalIntegrators::LDG::numericalFluxSigmaFromUBoundary
      (referenceDirection,
       dinfo.matrix(1,false),
       info.fe_values(0),
       info.fe_values(1) );

    elas::LocalIntegrators::LDG::U_MinusRHS_FromNueMinus
      (referenceDirection,
       info.fe_values(1),
       dinfo.vector(0).block(1)  );
    
    break;
  default:
    Assert(false, dealii::ExcInternalError() );
    break;
  
  }

  
}

} //End namespace LDGIntegrator
} //End namespace elas

template class elas::LDGIntegrator::LDGIntegrator<2>;
template class elas::LDGIntegrator::LDGIntegrator<3>;
