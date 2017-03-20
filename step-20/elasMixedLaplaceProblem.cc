#include "elasMixedLaplaceProblem.hpp"
#include "LDGIntegrator.hpp"
#include "LDGErrorIntegrator.hpp"
#include "utilities.hpp"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/schur_complement.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/matrix_block.h>
#include <deal.II/meshworker/assembler.h>

#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>


//#include "globals.hpp"



namespace elas
{

template <int dim>
elasMixedLaplaceProblem<dim>::elasMixedLaplaceProblem (const unsigned int degree)
  :
  degree (degree),
  fe (dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree), (dim*dim+dim)/2),1,
      dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree), dim), 1 ),
  dof_handler (triangulation),  
  EstimatedError(2),
  mapping(1),
  faceMapping(1),
  mg_SystemMatricesCollection(true,true)
{
  for(unsigned int i = 0; i < dim; ++i){
    referenceDirection[i] = 1.0;
  }

  referenceDirection[0] = -1.0;
  referenceDirection[1] = -1.0;
}




template<int dim>
void
elasMixedLaplaceProblem<dim>::EstimateError(){
  std::vector<unsigned int> old_user_indices;

  triangulation.save_user_indices(old_user_indices);
  EstimatedError.block(0).reinit(triangulation.n_active_cells() );
  EstimatedError.block(1).reinit(triangulation.n_active_cells() );
  //EstimatedError.collect_sizes();
  {
    unsigned int i = 0;
    for(auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell, ++i){
      cell->set_user_index(i);
    }
  }

  dealii::MeshWorker::IntegrationInfoBox<dim> info_boxSys;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

  info_boxSys.initialize_gauss_quadrature(n_gauss_points,
  					  n_gauss_points,
  					  n_gauss_points);
  
  dealii::AnyData SystemState;

  SystemState.add(&solution,"solution");

  info_boxSys.cell_selector.add( "solution", true, true, true);
  info_boxSys.boundary_selector.add("solution", true, true, true);
  info_boxSys.face_selector.add( "solution", true, true, true);

  dealii::UpdateFlags update_flags
    = dealii::update_values
    | dealii::update_quadrature_points;  

  info_boxSys.initialize_update_flags(true);
  info_boxSys.add_update_flags_all(update_flags);
  info_boxSys.add_update_flags_face(dealii::update_normal_vectors);
  info_boxSys.add_update_flags_boundary(dealii::update_normal_vectors);

  dealii::MeshWorker::DoFInfo<dim> dof_infoSys(dof_handler.block_info() );

  info_boxSys.initialize(fe, mapping, SystemState, solution, & dof_handler.block_info() );

  dealii::MeshWorker::Assembler::CellsAndFaces<elas::real> assemblerError;

  dealii::AnyData OutDataA;

  OutDataA.add(&EstimatedError, "cells");
  
  assemblerError.initialize(OutDataA, false);

   elas::LDGErrorIntegrator::LDGErrorIntegrator<dim>
     LDGErrorIntegrator;

   dealii::MeshWorker::LoopControl loopControl;
   loopControl.cells_first = false;

   dealii::MeshWorker::integration_loop<dim,dim>
     (dof_handler.begin_active(),
      dof_handler.end(),
      dof_infoSys,
      info_boxSys,
      LDGErrorIntegrator,
      assemblerError,
      loopControl);

   triangulation.load_user_indices(old_user_indices);
   
}

template <int dim>
void elasMixedLaplaceProblem<dim>::assign_boundary_ids()
{
  const dealii::types::boundary_id
    dir_plus = 1,
    dir_minus = 0,
    nue_plus = 3,
    nue_minus = 2;
  assert(2 == dim && "Only implemented for dim == 2" );
  dealii::QGauss<dim-1> face_quadrature_formula(1);
  const dealii::UpdateFlags face_update_flags =
    dealii::update_values | dealii::update_quadrature_points | dealii::update_normal_vectors;
  for(auto cell_sys = triangulation.begin_active();
      cell_sys != triangulation.end();
      ++cell_sys){
    for(unsigned int face_no = 0;
	face_no < dealii::GeometryInfo<dim>::faces_per_cell;
	++face_no){
      auto face_sys = cell_sys->face(face_no);

      if(face_sys->at_boundary() ){
	dealii::FEFaceValues<dim> fe_v_face_U(mapping, fe, face_quadrature_formula, face_update_flags);

	fe_v_face_U.reinit(cell_sys,face_no);
	const auto quad_point = fe_v_face_U.quadrature_point(0);
	const auto normal = fe_v_face_U.normal_vector(0);
	if(normal * referenceDirection > 0){
	  face_sys->set_boundary_id(nue_plus);	  
	} else {
	  if(std::abs(quad_point[0]) < 1e-10 || std::abs(quad_point[1]) < 1e-10) {
	    face_sys->set_boundary_id(nue_minus);
	  } else {
	    face_sys->set_boundary_id(dir_minus);	    
	  }
	}
      }
    }
  }      
}

template <int dim>
void
elasMixedLaplaceProblem<dim>::make_grid (){

  const dealii::types::boundary_id
    dir_plus = 1,
    dir_minus = 0,
    nue_plus = 3,
    nue_minus = 2;

  dealii::GridGenerator::hyper_L(triangulation);

  BoundaryIDMap[ dir_plus ] = elas::myBoundaryID::dir_Plus;
  BoundaryIDMap[ dir_minus ] = elas::myBoundaryID::dir_Minus;
  BoundaryIDMap[ nue_plus ] = elas::myBoundaryID::nue_Plus;
  BoundaryIDMap[ nue_minus ] = elas::myBoundaryID::nue_Minus;

}

template<int dim>
void
elasMixedLaplaceProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();
  
  
  dealii::DoFRenumbering::downstream(dof_handler,
				     (dealii::Point<dim>)referenceDirection,
				     false);
  dealii::DoFRenumbering::block_wise (dof_handler);    
    

  dealii::deallog << "   Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << " (by level: ";
  for (unsigned int level=0; level<triangulation.n_levels(); ++level){
    dealii::deallog << dof_handler.n_dofs(level)
	    << (level == triangulation.n_levels()-1
                  ? ")" : ", ");
    dealii::deallog << std::endl;
  }

  std::vector<dealii::types::global_dof_index>
    dofs_per_component ((dim*dim+dim)/2+dim);
  
  dealii::DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

  const unsigned int
    n_sigma = (dim*dim+dim)/2*dofs_per_component[0],
    n_u = dim*dofs_per_component[1];
  
  std::cout << "Number of active cells: "
	    << triangulation.n_active_cells()
	    << std::endl
	    << "Total number of cells: "
	    << triangulation.n_cells()
	    << std::endl
	    << "Number of degrees of freedom: "
	    << dof_handler.n_dofs()
	    << " (" << n_sigma << '+' << n_u << ')'
	    << std::endl;
  
  dealii::Table<2, dealii::DoFTools::Coupling>
    interiorMask(fe.n_components(),fe.n_components() );
    
  dealii::Table<2,dealii::DoFTools::Coupling>
    fluxMask(fe.n_components(), fe.n_components() );

  for(unsigned int c = 0; c< fe.n_components(); ++c){
    for(unsigned int d = 0; d < fe.n_components(); ++d){
      interiorMask[c][d]= dealii::DoFTools::none;
      fluxMask[c][d]= dealii::DoFTools::none;
    }
  }

  //sigma and sigma
  for(int i = 0; i < (dim * dim + dim ) / 2; ++i){
    for(int j = 0; j < (dim * dim+dim) / 2; ++j){	
      interiorMask[i][j] = dealii::DoFTools::nonzero;	
    }
  }

    
  const unsigned int offset = (dim * dim + dim)/2;

  //u and u
  for(int i = 0; i < dim; ++i){
    for(int j = 0; j < dim; ++j){
      interiorMask[i+offset][j+offset] = dealii::DoFTools::nonzero;
    }
  }

  //(u and sigma) AND (sigma and u)
  for(int i = 0; i < (dim * dim + dim )/2; ++i){
    for(int j = 0; j < dim; ++j){
      interiorMask[i][j+offset] = dealii::DoFTools::nonzero;
      fluxMask[i][j+offset] = dealii::DoFTools::nonzero;
      interiorMask[j+offset][i] = dealii::DoFTools::nonzero;
      fluxMask[j+offset][i] = dealii::DoFTools::nonzero;
    }
  }
     
  dealii::BlockDynamicSparsityPattern dsp(2,2);
  dsp.block(0,0).reinit(n_sigma,n_sigma);
  dsp.block(0,1).reinit(n_sigma,n_u);
  dsp.block(1,0).reinit(n_u,n_sigma);
  dsp.block(1,1).reinit(n_u,n_u);
    
  dsp.collect_sizes();
    
  dealii::DoFTools::make_flux_sparsity_pattern(dof_handler,dsp, interiorMask, fluxMask);
    
  sparsity_pattern.copy_from(dsp);
  sparsity_pattern.compress();
  //system_matrix.reinit (sparsity_pattern);
    
  //assert(fe.n_components() == (dim*dim+dim)/2 + dim );  

  

  

  solution.reinit (2);
  solution.block(0).reinit (n_sigma);
  solution.block(1).reinit (n_u);
  solution.collect_sizes ();

  system_rhs.reinit (2);
  system_rhs.block(0).reinit (n_sigma);
  system_rhs.block(1).reinit (n_u);
  system_rhs.collect_sizes ();

  blockConstraints1.reinit(2);
  blockConstraints1.block(0).reinit(n_sigma);
  blockConstraints1.block(1).reinit(n_u);
  blockConstraints1.collect_sizes();
  
  SystemMatricesCollection.clear(false);
  //mg_SystemMatricesCollection.clear(false);

  SystemMatricesCollection.add(0,0,"MassSigma"); // 0
  //  mg_SystemMatricesCollection.add(0,0,"MassSigma"); // 0

  SystemMatricesCollection.add(0,1,"StiffSigmaFromU"); //1
  //mg_SystemMatricesCollection.add(0,1,"StiffSigmaFromU"); //1

  SystemMatricesCollection.add(1,0,"StiffUFromSigma"); //2
  //mg_SystemMatricesCollection.add(1,0,"StiffUFromSigma"); //2
  
  SystemMatricesCollection.add(1,1,"BoundaryMassForDirMinus");  //3
  //mg_SystemMatricesCollection.add(1,1,"BoundaryMassForDirMinus");  //3
  
  SystemMatricesCollection.add(0,0,"BoundaryMassForNuePlus");  //4
  //mg_SystemMatricesCollection.add(0,0,"BoundaryMassForNuePlus");  //4
  
  SystemMatricesCollection.add(0,0,"InverseMassSigma");        //5
  //mg_SystemMatricesCollection.add(0,0,"InverseMassSigma");        //5
  
  SystemMatricesCollection.add(1,1,"SvdU_ForDirMinus");        //6
  //mg_SystemMatricesCollection.add(1,1,"SvdU_ForDirMinus");        //6
  
  SystemMatricesCollection.add(1,1,"Svd_CapitalSigma_ForDirMinus");  //7
  //mg_SystemMatricesCollection.add(1,1,"Svd_CapitalSigma_ForDirMinus");  //7

  SystemMatricesCollection.add(0,0,"SvdU_ForNuePlus"); //8
  //mg_SystemMatricesCollection.add(0,0,"SvdU_ForNuePlus"); //8
  
  SystemMatricesCollection.add(0,0,"Svd_CapitalSigma_ForNuePlus"); //9
  //mg_SystemMatricesCollection.add(0,0,"Svd_CapitalSigma_ForNuePlus"); //9

  SystemMatricesCollection.add(0,0,"SvdV_ForNuePlus"); //10
  //mg_SystemMatricesCollection.add(0,0,"SvdV_ForNuePlus"); //10

  SystemMatricesCollection.add(1,1,"SvdV_ForDirMinus"); //11
  //mg_SystemMatricesCollection.add(1,1,"SvdV_ForDirMinus"); //11
  
  SystemMatricesCollection.reinit(sparsity_pattern);

  SystemRHSsAndConstraints.add(&system_rhs, "system_rhs");
  SystemRHSsAndConstraints.add(&blockConstraints1, "BlockConstraints1");

  const unsigned int n_levels=triangulation.n_levels();
  //mg_SystemMatricesCollection.resize(0,n_levels-1);
  for(unsigned int level = 0; level < n_levels; ++level){
      dealii::BlockDynamicSparsityPattern dsp_level(2,2);
      dsp_level.block(0,0).reinit(n_sigma,n_sigma);
      dsp_level.block(0,1).reinit(n_sigma,n_u);
      dsp_level.block(1,0).reinit(n_u,n_sigma);
      dsp_level.block(1,1).reinit(n_u,n_u);
    
      dsp_level.collect_sizes();
    
      dealii::
	MGTools::
	make_flux_sparsity_pattern(dof_handler,
				   dsp_level,
				   level,
				   interiorMask,
				   fluxMask);

      

      //mg_sparsity_patterns[level].copy_from(dsp);

      //mg_SystemMatricesCollection[level].reinit_matrix(mg_sparsity_patterns[level]);

      
  
    
  }
  //mg_SystemMatricesCollection.reinit_matrix(mg_sparsity_patterns);
    
}



template <int dim>
void elasMixedLaplaceProblem<dim>::assemble_system_PreProcess ()
{
  system_matrix.reinit(sparsity_pattern);

  massSigma.reinit( sparsity_pattern.block(0,0) );
  Svd_U_for_Nue.reinit(sparsity_pattern.block(0,0) );
  Svd_Sigma_for_Nue.reinit(sparsity_pattern.block(0,0) );
  BoundaryMassSigma.reinit(sparsity_pattern.block(0,0) );
  InverseMassSigma.reinit(sparsity_pattern.block(0,0) );
  StiffSigmaFromU.reinit(sparsity_pattern.block(0,1) );
  

  MassU.reinit(sparsity_pattern.block(1,1));
  Svd_U_for_Dir.reinit(sparsity_pattern.block(1,1) );
  Svd_Sigma_for_Dir.reinit(sparsity_pattern.block(1,1) );
  InverseMassU.reinit(sparsity_pattern.block(1,1));
  BoundaryMassU.reinit(sparsity_pattern.block(1,1));
  StiffUFromSigma.reinit(sparsity_pattern.block(1,0) );    	       
}

template <int dim>
void
elasMixedLaplaceProblem<dim>::assemble_system_mg()
{
    //constraints;

  dealii::MeshWorker::IntegrationInfoBox<dim> info_boxSys;
  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

  info_boxSys.initialize_gauss_quadrature(n_gauss_points,
					  n_gauss_points,
					  n_gauss_points);  

  dealii::UpdateFlags update_flags
    = dealii::update_values
    | dealii::update_gradients
    | dealii::update_quadrature_points;


  info_boxSys.initialize_update_flags(true);

  info_boxSys.initialize_update_flags(update_flags);
  info_boxSys.add_update_flags_all(update_flags);
  info_boxSys.add_update_flags_face(dealii::update_normal_vectors);
  info_boxSys.add_update_flags_boundary(dealii::update_normal_vectors);
  info_boxSys.add_update_flags_cell(dealii::update_quadrature_points);
  info_boxSys.add_update_flags_cell(dealii::update_values);

  dealii::MeshWorker::DoFInfo<dim> dof_InfoSys(dof_handler.block_info() );

  info_boxSys.initialize(fe, mapping, &dof_handler.block_info() );
  
  
  dealii
    ::MeshWorker
    ::Assembler
    ::MGMatrixLocalBlocksToGlobalBlocks<dealii::SparseMatrix<elas::real>,
					elas::real >
    assembleSystem( std::numeric_limits<elas::real>::min() );

  assembleSystem.initialize(&(dof_handler.block_info()),
			    mg_SystemMatricesCollection);
			    
  

  elas::LDGIntegrator::LDGIntegrator<dim> LDGintegrator(referenceDirection,
                                                        info_boxSys.face_quadrature,
                                                        info_boxSys.face_flags,
                                                        mapping,
                                                        &BoundaryIDMap);

  
  // //Dealii::MeshWorker::LoopControl loopControl;


  dealii::MeshWorker::integration_loop<dim,dim>
    (dof_handler.begin_mg(),
     dof_handler.end_mg(),
     dof_InfoSys,
     info_boxSys,
     LDGintegrator,
     assembleSystem);
    

}
  

template <int dim>
void elasMixedLaplaceProblem<dim>::assemble_system()
{
  //constraints;

  dealii::MeshWorker::IntegrationInfoBox<dim> info_boxSys;
  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

  info_boxSys.initialize_gauss_quadrature(n_gauss_points,
					  n_gauss_points,
					  n_gauss_points);  

  dealii::UpdateFlags update_flags
    = dealii::update_values
    | dealii::update_gradients
    | dealii::update_quadrature_points;


  info_boxSys.initialize_update_flags(true);

  info_boxSys.initialize_update_flags(update_flags);
  info_boxSys.add_update_flags_all(update_flags);
  info_boxSys.add_update_flags_face(dealii::update_normal_vectors);
  info_boxSys.add_update_flags_boundary(dealii::update_normal_vectors);
  info_boxSys.add_update_flags_cell(dealii::update_quadrature_points);
  info_boxSys.add_update_flags_cell(dealii::update_values);

  dealii::MeshWorker::DoFInfo<dim> dof_InfoSys(dof_handler.block_info() );

  info_boxSys.initialize(fe, mapping, &dof_handler.block_info() );  
  
  dealii
    ::MeshWorker
    ::Assembler
    ::SystemLocalBlocksToGlobalBlocks<dealii::SparseMatrix<elas::real>,
				      dealii::BlockVector<elas::real>,
				      elas::real >
    assembleSystem( std::numeric_limits<double>::min() );

  assembleSystem.initialize(&(dof_handler.block_info() ),
			    SystemMatricesCollection,
			    SystemRHSsAndConstraints);
  

  elas::LDGIntegrator::LDGIntegrator<dim> LDGintegrator(referenceDirection,
                                                        info_boxSys.face_quadrature,
                                                        info_boxSys.face_flags,
                                                        mapping,
                                                        &BoundaryIDMap);


  dealii::MeshWorker::LoopControl loopControl;


  dealii::MeshWorker::integration_loop<dim,dim>
    (dof_handler.begin_active(),
     dof_handler.end(),
     dof_InfoSys,
     info_boxSys,
     LDGintegrator,
     assembleSystem,
     loopControl);

}

template <int dim>
void
elasMixedLaplaceProblem<dim>::assemble_system_PostProcess()
{
  elas::real zero = 0.0;
  unsigned int n_dofs_U = 0, n_dofs_sigma = 0;
  {    
    std::vector<dealii::types::global_dof_index> dofs_per_component ((dim*dim+dim)/2+dim);
    dealii::DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    
    n_dofs_sigma = ((dim*dim+dim)/2) * dofs_per_component[0];
    n_dofs_U = dim*dofs_per_component[1];
  }
  
  {
    //unsigned int
      rankBoundaryMassSigma = 0,
      nullityBoundaryMassSigma= 0;
    
    std::vector<unsigned int> Prows, Qrows;

    for(unsigned int i = 0; i < SystemMatricesCollection.matrix(9).m(); ++i){
      if(fabs( SystemMatricesCollection.matrix(9)(i,i) ) > zero ){
	++rankBoundaryMassSigma;
	Prows.emplace_back(i);
      }
      else{
	++nullityBoundaryMassSigma;
	Qrows.emplace_back(i);
      }
    }    
    
    PP.reinit(rankBoundaryMassSigma, n_dofs_sigma);
    QQ.reinit(nullityBoundaryMassSigma, n_dofs_sigma);

    PP = 0;
    QQ = 0;

    for(unsigned int i = 0; i < rankBoundaryMassSigma; ++i){
      PP.add(i,Prows[i],1.0);    
    }    
    
    for(unsigned int i = 0; i < nullityBoundaryMassSigma; ++i){
      QQ.add(i,Qrows[i],1.0);
    }    
  }
  
  { 
    //unsigned int
      rankBoundaryMassU = 0,
      nullityBoundaryMassU = 0;
   
   std::vector<unsigned int> Rrows, Srows;

   
   for(unsigned int i = 0; i < SystemMatricesCollection.matrix(7).m(); ++i){          
     if(fabs( SystemMatricesCollection.matrix(7)(i,i) ) > zero){
       ++rankBoundaryMassU;
       Rrows.emplace_back(i);
     }
     else{
       ++nullityBoundaryMassU;
       Srows.emplace_back(i);
     }
   }

   RR = 0;
   SS = 0;
   RR.reinit(rankBoundaryMassU, n_dofs_U);
   SS.reinit(nullityBoundaryMassU, n_dofs_U);

   for(unsigned int i = 0; i < rankBoundaryMassU; ++i){
     RR.add(i,Rrows[i],1.0);      
   }

   for(unsigned int i = 0; i < nullityBoundaryMassU; ++i){
     SS.add(i,Srows[i],1.0);
   }    
  }
  sparsity_pattern.compress();
}

template <int dim>
void elasMixedLaplaceProblem<dim>::distill_matrices()
{
  //dealii::SparsityPattern
  //massSigmaSP;
  DistillMatrix(SystemMatricesCollection.matrix(1), StiffSigmaFromU_SP);
  DistillMatrix(SystemMatricesCollection.matrix(2), StiffUFromSigma_SP);
  DistillMatrix(SystemMatricesCollection.matrix(0), massSigma_SP);
  DistillMatrix(SystemMatricesCollection.matrix(5), inverseMassSigma_SP);
  DistillMatrix(SystemMatricesCollection.matrix(3), BoundaryMassUForDirMinus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(4), BoundaryMassQForNuePlus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(6), SvdU_ForDirMinus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(8), SvdU_ForNuePlus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(9), Svd_CapitalSigma_ForNuePlus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(7), Svd_CapitalSigma_ForDirMinus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(10), SvdV_ForNuePlus_SP);
  DistillMatrix(SystemMatricesCollection.matrix(11), SvdV_ForDirMinus_SP);  
  
  
}

template <int dim>
void elasMixedLaplaceProblem<dim>::solve ()
{
  //We define some references to so we can save some writing;
  const auto 
    svdUForNue = &(SystemMatricesCollection.matrix(8)),
    svdSigmaForNue = &(SystemMatricesCollection.matrix(9)),
    svdUForDir = &(SystemMatricesCollection.matrix(6)),
    svdSigmaForDir = &(SystemMatricesCollection.matrix(7)),
    MassSigma = &(SystemMatricesCollection.matrix(0)),
    StiffSigmaFromU = &(SystemMatricesCollection.matrix(1)),
    StiffUFromSigma = &(SystemMatricesCollection.matrix(2)),
    InverseMassSigma = &(SystemMatricesCollection.matrix(5));

  const auto
    scriptQ =
    dealii::linear_operator(*svdUForNue)
    * dealii::transpose_operator(dealii::linear_operator(QQ)),
      
    scriptP =
    dealii::linear_operator(*svdUForNue)
    * dealii::transpose_operator(dealii::linear_operator(PP)),

    scriptR =
    dealii::linear_operator(*svdUForDir)
    * dealii::transpose_operator(dealii::linear_operator(RR)),    

    scriptS =
    dealii::linear_operator(*svdUForDir)
    * dealii::transpose_operator(dealii::linear_operator(SS));
  
  const auto
    PSigmaPStar
    = dealii::linear_operator(PP)
    * dealii::linear_operator(*svdSigmaForNue)
    * dealii::transpose_operator( dealii::linear_operator(PP) );

  auto Id = dealii::PreconditionIdentity();
  dealii::SolverControl firstSolverControl(100,1e-14);
  dealii::SolverCG<dealii::Vector<elas::real> >
    firstSolver(firstSolverControl);

  const auto PPSigmaPPStar_Inverse
    = dealii::inverse_operator(PSigmaPStar,
			       firstSolver,
			       Id);

  dealii::Vector<elas::real>  scriptm;
  //find script lower case m;  
  if(rankBoundaryMassSigma > 0){    

    scriptm.reinit(rankBoundaryMassSigma);
    dealii::Vector<elas::real> localRHS( PP.m() );

    dealii::transpose_operator(scriptP).vmult
      (localRHS,
       SystemRHSsAndConstraints.entry<dealii::BlockVector<elas::real>* >(1)->block(0));

    
    PPSigmaPPStar_Inverse.vmult(scriptm,localRHS);
    {
      std::ofstream strm("scriptm.csv");
      scriptm.print(strm);
    }

  }

  const auto
    RRSigmaRRStar
    = dealii::linear_operator(RR)
    * dealii::linear_operator(*svdSigmaForDir)
    * dealii::transpose_operator( dealii::linear_operator(RR ));

  auto Id2 = dealii::PreconditionIdentity();
  dealii::SolverControl secondSolverControl(100,1e-14);
  dealii::SolverCG<dealii::Vector<elas::real> > secondSolver(secondSolverControl);

  const auto RRSigmaRRStar_Inverse
    = dealii::inverse_operator(RRSigmaRRStar,
			       secondSolver,
			       Id2);  
  
  dealii::Vector<elas::real>  scriptl;
  //find lowercase script l
  if(rankBoundaryMassU > 0){
    scriptl.reinit(rankBoundaryMassU);
    dealii::Vector<elas::real> localRHS( RR.m() );
    (dealii::transpose_operator(scriptR)).vmult
      (localRHS,
       SystemRHSsAndConstraints.entry<dealii::BlockVector<elas::real> * >(1)->block(1));
    

    RRSigmaRRStar_Inverse.vmult(scriptl, localRHS);

    {
      std::ofstream strm("scriptl.csv");
      scriptl.print(strm);
    }

    

  }

  

  //dealii::Vector<elas::real> doubleStruckF (scriptQ.n() );
  //make double struck F
  dealii::Vector<elas::real> doubleStruckF (QQ.m() );


  


  dealii::transpose_operator(scriptQ).vmult(doubleStruckF,
					    SystemRHSsAndConstraints.
					    entry<dealii::BlockVector<elas::real>* >(0)->block(0));



  // for(unsigned int i = 0; i < 	  SystemRHSsAndConstraints.
  // 	entry<dealii::BlockVector<elas::real>* >(0)->block(0).size(); ++i){
  //   PRINT(SystemRHSsAndConstraints.
  // 	  entry<dealii::BlockVector<elas::real>* >(0)->block(0)[i] );
  // }


  // for(unsigned int i = 0; i < doubleStruckF.size(); ++i){
  //   PRINT(doubleStruckF[i]);
  // }
  // assert(false);
  
  if(rankBoundaryMassSigma > 0){

    const auto product1 =
      dealii::transpose_operator(scriptQ)
      * dealii::linear_operator(*MassSigma)
      * scriptP;
    product1.vmult_add(doubleStruckF, scriptm);

  }
  if(rankBoundaryMassU > 0){
    const auto product2
      = dealii::transpose_operator(scriptQ)
      * dealii::linear_operator(*StiffSigmaFromU)
      * scriptR;
    product2.vmult_add(doubleStruckF, scriptl);
  }

  {
    std::ofstream strm("doubleStruckF.csv");
    doubleStruckF.print(strm,16);
  }

  
  //make double struck G
  dealii::Vector<elas::real> doubleStruckG (SS.m() ) ;
  dealii::transpose_operator(scriptS).
    vmult(doubleStruckG,
	  SystemRHSsAndConstraints.
	  entry<dealii::BlockVector<elas::real>* >(0)->block(1));  

  if(rankBoundaryMassU > 0){    
    const auto product
      = dealii::transpose_operator(scriptS)
      * dealii::linear_operator(*StiffUFromSigma )
      * scriptP;

    product.vmult_add(doubleStruckG, scriptm);
  }

  {
    std::ofstream strm("doubleStruckG.csv");
    doubleStruckG.print(strm,16);
  }

  
    
  //make script capital A;
  const auto
    QstarMQ =
    dealii::transpose_operator(scriptQ)
    * dealii::linear_operator(*MassSigma)
    * scriptQ;

  const auto
    QstarMQ_ApproxInverse =
    dealii::transpose_operator(scriptQ)
    * dealii::linear_operator(*InverseMassSigma)
    * scriptQ;

  dealii::SolverControl InnerControlForQstarMQ(1000, 1e-14);
  dealii::SolverCG<> innerSolver(InnerControlForQstarMQ);

  const auto
    QstarMQ_Inverse
    = dealii::inverse_operator(QstarMQ, innerSolver, QstarMQ_ApproxInverse);


  const auto
   scriptB
    = dealii::transpose_operator(scriptQ)
    * dealii::linear_operator(*StiffSigmaFromU)
    * scriptS;

  const auto
    minusScriptBStar
    = dealii::transpose_operator(scriptS)
    * dealii::linear_operator(*StiffUFromSigma)
    * scriptQ;

  const auto
    scriptA_Approx
    = dealii::transpose_operator(scriptB)
    * QstarMQ_ApproxInverse
    * scriptB;

  const auto
    scriptA_Approx_TEMP
    = dealii::transpose_operator(scriptS)
    * dealii::transpose_operator(dealii::linear_operator(*StiffSigmaFromU))
    * dealii::linear_operator(*InverseMassSigma)
    * dealii::linear_operator(*StiffSigmaFromU)
    * scriptS;

  const auto
    stiffSigmaFromU_Oper = dealii::linear_operator(*StiffSigmaFromU);
  
  const auto scriptZ 
    = dealii::transpose_operator(dealii::linear_operator(*StiffSigmaFromU))
    * dealii::linear_operator(*InverseMassSigma)
    * dealii::linear_operator(*StiffSigmaFromU);

  dealii::SolverControl InnerControlForScriptZ(1000, 1e-12);
  dealii::SolverCG<> scriptZSolver(InnerControlForScriptZ);
  const auto ID = dealii::PreconditionIdentity();
  const auto scriptZ_inverse = dealii::inverse_operator(scriptZ, scriptZSolver, ID);

  const auto
    scriptA_Approx_TEMP_Pre
    = dealii::transpose_operator(scriptS)
    * scriptZ_inverse
    * scriptS;

  dealii::SolverControl InnerControlForScriptA_Approx(1000,1e-10);
  dealii::SolverCG<> scriptA_ApproxSolver(InnerControlForScriptA_Approx);
  const auto scriptA_Approx_TEMP_Inverse = dealii::inverse_operator(scriptA_Approx_TEMP,
								    scriptA_ApproxSolver,
								    scriptA_Approx_TEMP_Pre);



  
  
  dealii::SolverControl InnerControlForPrecon(1000, 1e-12);
  
  dealii::SolverCG<> PreconSolver(InnerControlForPrecon);  
  
  
  
  //A pretty good approximation of the inverse is the inverse of the approximation.
  const auto scriptA_inverse_Approx = dealii::inverse_operator(scriptA_Approx, PreconSolver, ID);

  

  dealii::SolverControl outerControl(1000, 1e-10);

  dealii::SolverCG<> outerSolver(outerControl);

  const auto dummy = dealii::transpose_operator(scriptS) * scriptS;

  const auto scriptD = dealii::null_operator(dummy);

  const auto scriptA= dealii::schur_complement(QstarMQ_Inverse, scriptB, minusScriptBStar, scriptD);

  const auto scriptA_inverse = dealii::inverse_operator(scriptA,outerSolver,ID);

  doubleStruckF *= -1;
  doubleStruckG *= -1;
  auto rhs = dealii::condense_schur_rhs(QstarMQ_Inverse, minusScriptBStar, doubleStruckF, doubleStruckG);

  const dealii::Vector<elas::real> alpha = scriptA_inverse * rhs;
  {
    std::ofstream strm("alpha.csv");
    alpha.print(strm,16);
  }

  const dealii::Vector<elas::real> beta
    = dealii::postprocess_schur_solution(QstarMQ_Inverse, scriptB, alpha, doubleStruckF);

  {
    std::ofstream strm("beta.csv");
    beta.print(strm,16);
  }

  //Populate solution;
  scriptR.vmult(solution.block(1), scriptl);
  scriptS.vmult_add(solution.block(1), alpha);
  
  scriptP.vmult(solution.block(0), scriptm);
  scriptQ.vmult_add(solution.block(0), beta);

  {
    std::ofstream strm("soln0.csv");
    solution.block(0).print(strm,16);
  }
  {
    std::ofstream strm("soln1.csv");
    solution.block(1).print(strm,16);
  }
    
}


  
  

  
  



  



//}


template <int dim>
void elasMixedLaplaceProblem<dim>::output_results () const
{
  typedef dealii::DataComponentInterpretation::DataComponentInterpretation DCI_type;
  std::vector<std::string> SolutionNames;
  std::vector<DCI_type> DataComponentInterpretation;

  for(unsigned int i = 0; i < (dim*dim + dim)/2; ++i){
    std::string name = "Stress";
    name += dealii::Utilities::int_to_string(i,1);
    SolutionNames.emplace_back(name);;
    DataComponentInterpretation.
      emplace_back
      (dealii::DataComponentInterpretation::component_is_scalar);
  }

  for(unsigned int i = 0; i < dim; ++i){
    SolutionNames.emplace_back("Displacement");
    DataComponentInterpretation.emplace_back(dealii::DataComponentInterpretation::component_is_part_of_vector);
  }    

  dealii::DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, SolutionNames,
			    dealii::DataOut<dim>::type_dof_data,
			    DataComponentInterpretation);

  data_out.build_patches (2*degree+1);

  std::ofstream output ("solution.vtk");
  data_out.write_vtk (output);
}

template <int dim>
void elasMixedLaplaceProblem<dim>::run ()
{
  for(unsigned int cycle = 0; cycle < 2; ++cycle){
    std::cout << "Cycle " << cycle << std::endl;
    if(0 == cycle){
      make_grid();      
      assign_boundary_ids();      
    }
    else {
      refine_grid();
    }  

    setup_system();
    assemble_system_PreProcess();
    assemble_system();
    assemble_system_mg();
  
    assemble_system_PostProcess();  
  
    distill_matrices();
  
    //TODO:  All of this printing should be moved 
    {
    
    for(unsigned int i = 0; i < SystemMatricesCollection.size(); ++i){
      std::string name = SystemMatricesCollection.name(i);

      std::cout << name
		<< ".n_nonzero_elements() = "
		<< SystemMatricesCollection.matrix(i).n_nonzero_elements()	
		<< std::endl;

      std::cout << name
		<< ".n_actually_nonzero_elements() = "
		<< SystemMatricesCollection.matrix(i).n_actually_nonzero_elements()	
		<< std::endl << std::endl;      
      name += std::to_string(cycle);
      name += ".mtx";
      
      std::ofstream ostr(name);
      PrintMatrixMarket(SystemMatricesCollection.matrix(i), ostr);      
    }    
  }

  {
    for(unsigned int i = 0; i < SystemRHSsAndConstraints.size(); ++i){
      
      const std::string name = SystemRHSsAndConstraints.name(i);
      std::cout << "name = " << name << std::endl << std::endl << std::endl;
      for(int j = 0; j < 2; ++j){
	const std::string number = std::to_string(j);
	const std::string extension = ".csv";
	const std::string namePlusNumber = name + number+extension;
	std::ofstream stream(namePlusNumber);
	SystemRHSsAndConstraints.entry<dealii::BlockVector<elas::real> *>(i)->block(j).print(stream,16);
      }
    }
  }

  solve();
  output_results ();
  EstimateError ();
  
  }
}

template<int dim>
void
elasMixedLaplaceProblem<dim>::refine_grid()
{  
  triangulation.refine_global(1);
}

} //end namespace elas;

template class elas::elasMixedLaplaceProblem<2>;
template class elas::elasMixedLaplaceProblem<3>;
