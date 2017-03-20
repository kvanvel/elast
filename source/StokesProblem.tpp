#ifndef STOKESPROBLEM_T_
#define STOKESPROBLEM_T_

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/multigrid/mg_transfer.h>

#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/integration_info.h>

#include "LDGIntegrator.hpp"


// #include <deal.II/lac/linear_operator.h>
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/schur_complement.h>
// #include <deal.II/lac/iterative_inverse.h>
// #include <deal.II/lac/packaged_operation.h>
// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/dofs/dof_renumbering.h>
// #include <deal.II/fe/fe_dgq.h>
// #include <deal.II/numerics/data_out.h>



namespace stokes {

template<int dim>
StokesProblem<dim>::StokesProblem(const unsigned int degree_)
  : degree(degree_)
  , triangulation (dealii::Triangulation<dim>::limit_level_difference_at_vertices)
  , fe( dealii::FESystem<dim>(dealii::FE_DGP<dim>(degree), ( dim * dim + dim) / 2) ,1
	,dealii::FESystem<dim>(dealii::FE_DGP<dim>(degree), dim),1
	,dealii::FESystem<dim>(dealii::FE_DGP<dim>(degree), 1), 1)
  , dof_handler (triangulation)
    
  , mapping(1)
{
  //TODO: Initializer list?
  for(unsigned int i = 0; i < dim; ++i){
    referenceDirection[i] = 1.0;
  }
}


template<int dim>
void
StokesProblem<dim>::run ()
{
  make_grid_and_dofs();
  assign_boundary_ids();
  assemble_system_PreProcess();
  assemble_system();
  print_matrices();
  //assemble_system_PostProcess();

  // distill_matrices();

  // solve();
  // output_results();
  // EstimateError ();
  
}

template<int dim>
void
StokesProblem<dim>::make_grid_and_dofs()  
{
  

  dealii::GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(1);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();

  dealii::DoFRenumbering::block_wise(dof_handler);
  std::vector<dealii::types::global_dof_index> dofs_per_component((dim*dim + dim)/2 + dim + 1);
  dealii::DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);

  const unsigned int
    n_sigma = (dim * dim + dim)/2 * dofs_per_component[0],
    n_U = dim * dofs_per_component[1],
    n_P = dofs_per_component[2];

  std::cout << "Number of active cells: "
	    << triangulation.n_active_cells()
	    << std::endl
	    << "Total number of cells: "
	    << triangulation.n_cells()
	    << std::endl
	    << "Number of degrees of freedom: "
	    << dof_handler.n_dofs()
	    << " = (" << n_sigma <<  " + " << n_U << " + " << n_P <<  " ) " << std::endl;

  {
    dealii::Table<2, dealii::DoFTools::Coupling>
      interiorMask(fe.n_components(), fe.n_components() );

    dealii::Table<2, dealii::DoFTools::Coupling>
      fluxMask(fe.n_components(), fe.n_components() );

    for(unsigned int c = 0; c < fe.n_components(); ++c){
      for(unsigned int d = 0; d < fe.n_components(); ++d){
	interiorMask[c][d] = dealii::DoFTools::none;
	fluxMask[c][d] = dealii::DoFTools::none;
      }
    }

    const int numComponentsSigma = (dim * dim + dim )/2;

    //sigma and sigma
    for(int i = 0; i < numComponentsSigma; ++i){
      for(int j = 0; j < numComponentsSigma; ++j){
	interiorMask[i][j] = dealii::DoFTools::nonzero;
      }
    }

    //(sigma and u) along with (u and sigma);
    for(int i = 0; i < numComponentsSigma; ++i){
      for(int j = 0; j < dim; ++j){
	interiorMask[i][j+numComponentsSigma] = dealii::DoFTools::nonzero;
	fluxMask[i][j+numComponentsSigma] = dealii::DoFTools::nonzero;
	interiorMask[j+numComponentsSigma][i] = dealii::DoFTools::nonzero;
	fluxMask[j+numComponentsSigma][i] = dealii::DoFTools::nonzero;
      }
    }

    const int offset = numComponentsSigma + dim;
    for(int i = 0; i < dim; ++i){
      for(int j = 0; j < 1; ++j){
	interiorMask[i][j+offset] = dealii::DoFTools::nonzero;
	fluxMask[i][j+offset] = dealii::DoFTools::nonzero;
	interiorMask[j+offset][i] = dealii::DoFTools::nonzero;
	fluxMask[j+offset][i] = dealii::DoFTools::nonzero;
      }
    }


    dealii::BlockDynamicSparsityPattern dsp(3,3);
    dsp.block(0,0).reinit(n_sigma,n_sigma);
    dsp.block(0,1).reinit(n_sigma,n_U);
    dsp.block(0,2).reinit(n_sigma,n_P);
    dsp.block(1,0).reinit(n_U,n_sigma);
    dsp.block(1,1).reinit(n_U,n_U);
    dsp.block(1,2).reinit(n_U,n_P);
    dsp.block(2,0).reinit(n_P,n_sigma);
    dsp.block(2,1).reinit(n_P,n_U);
    dsp.block(2,2).reinit(n_P,n_P);
    
    dsp.collect_sizes();

    //dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, interiorMask,fluxMask);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    
  }

  const unsigned int
    n_couplings = 0;

  assert(fe.n_components() == (dim * dim + dim)/2 + dim + 1 );

  sparsity_pattern.compress();
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(3);
  solution.block(0).reinit(n_sigma);
  solution.block(1).reinit(n_U);
  solution.block(2).reinit(n_P);
  solution.collect_sizes();

  system_rhs.reinit(3);
  system_rhs.block(0).reinit(n_sigma);
  system_rhs.block(1).reinit(n_U);
  system_rhs.block(2).reinit(n_P);

  SystemMatricesCollection.clear(false);
  SystemMatricesCollection.add(0,0,"MassSigma");        // 0 
  SystemMatricesCollection.add(0,0,"InverseMassSigma"); // 1
  SystemMatricesCollection.add(0,1,"StiffSigmaFromU");  // 2
  SystemMatricesCollection.add(1,0,"StiffUFromSigma");  // 3  
  SystemMatricesCollection.add(1,2,"StiffUFromP");      // 4
  SystemMatricesCollection.add(2,1,"StiffPFromU");      // 5

  SystemMatricesCollection.reinit(sparsity_pattern);
    
}

template <int dim>
void
StokesProblem<dim>::assign_boundary_ids()
{
  const dealii::types::boundary_id
    dir_plus = 1,
    dir_minus = 0,
    nue_plus = 3,
    nue_minus = 2;

  BoundaryIDMap[ dir_plus ] = stokes::myBoundaryID::dir_Plus;
  BoundaryIDMap[ dir_minus ] = stokes::myBoundaryID::dir_Minus;
  BoundaryIDMap[ nue_plus ] = stokes::myBoundaryID::nue_Plus;
  BoundaryIDMap[ nue_minus ] = stokes::myBoundaryID::nue_Minus;
  
  dealii::QGauss<dim-1> face_quadrature_formula(1);
  const dealii::UpdateFlags face_update_flags =
    dealii::update_values
    | dealii::update_quadrature_points
    | dealii::update_normal_vectors;
  for(auto cell_sys = triangulation.begin_active();
      cell_sys != triangulation.end();
      ++cell_sys){
    for(unsigned int face_no = 0;
	face_no < dealii::GeometryInfo<dim>::faces_per_cell;
	++face_no){
      auto face_sys = cell_sys->face(face_no);
      if(face_sys->at_boundary() ){
	dealii::FEFaceValues<dim> fe_v_face(mapping, fe, face_quadrature_formula, face_update_flags);
	fe_v_face.reinit(cell_sys, face_no);
	const auto quad_point = fe_v_face.quadrature_point(0);
	const auto normal = fe_v_face.normal_vector(0);
	if(normal * referenceDirection > stokes::real(0.0)){
	  face_sys->set_boundary_id(nue_plus);	  
	} else {
	  if( std::abs(quad_point[0]) < stokes::real(1.0e-10)
	      || std::abs(quad_point[1]) < stokes::real(1.0e-10) ){
	    face_sys->set_boundary_id(nue_minus);	    
	  }
	  else {
	    face_sys->set_boundary_id(dir_minus);	    
	  }
	}
      }
    }
  }
}

template <int dim>
void
StokesProblem<dim>::assemble_system_PreProcess ()
{
  system_matrix.reinit(sparsity_pattern);

  massSigma.reinit(        sparsity_pattern.block(0,0));
  inverseMassSigma.reinit( sparsity_pattern.block(0,0));
  stiffSigmaFromU.reinit(  sparsity_pattern.block(0,1));
  stiffUFromSigma.reinit(  sparsity_pattern.block(1,0));
  stiffUFromP.reinit(      sparsity_pattern.block(1,2));
  stiffPFromU.reinit(      sparsity_pattern.block(2,1));  
}

template <int dim>
void
StokesProblem<dim>::assemble_system()
{
  dealii::MeshWorker::IntegrationInfoBox<dim> info_boxSys;
  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;

  info_boxSys.initialize_gauss_quadrature(n_gauss_points,
					  n_gauss_points,
					  n_gauss_points);

  const dealii::UpdateFlags update_flags
    = dealii::update_values
    | dealii::update_gradients
    | dealii::update_quadrature_points;

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
    ::SystemLocalBlocksToGlobalBlocks<dealii::SparseMatrix<stokes::real>,
				      dealii::BlockVector<stokes::real>,
				      stokes::real>
    assembleSystem( std::numeric_limits<stokes::real>::min() );

  assembleSystem.initialize(&(dof_handler.block_info() ),
			    SystemMatricesCollection,
			    SystemRHSsAndConstraints);

  stokes
    ::LDGIntegrator
    ::LDGIntegrator<dim> LDGIntegrator(referenceDirection,
  				       info_boxSys.face_quadrature,
  				       info_boxSys.face_flags,
  				       mapping,
  				       BoundaryIDMap); 

  dealii::MeshWorker::LoopControl loopControl;

  dealii
    ::MeshWorker
    ::integration_loop<dim,dim> (dof_handler.begin_active(),
  				 dof_handler.end(),
  				 dof_InfoSys,
  				 info_boxSys,
  				 LDGIntegrator,
  				 assembleSystem,
  				 loopControl);
}

template <int dim>
void
StokesProblem<dim>::print_matrices ()
{
  for(unsigned int i = 0; i < SystemMatricesCollection.size(); ++i){
    std::string name = SystemMatricesCollection.name(i);
    name+= ".mtx";
    std::ofstream ostrm(name);
    PrintMatrixMarket(
		      SystemMatricesCollection.matrix(i),
		      ostrm,
		      1e-10);
  }

  std::vector<std::vector<dealii::FullMatrix<stokes::real> > > projection_matrices;
   // dealii::FETools::compute_projection_matrices(fe.base_element(2),
   // 						projection_matrices,
   // 						true);

  std::vector<std::vector<dealii::FullMatrix<stokes::real > > > embedding_matrices;
  // dealii::FETools::compute_embedding_matrices(fe.base_element(2),
  // 					      embedding_matrices,
  // 					      true);

  dealii::MGTransferPrebuilt<dealii::Vector<double> > mg_transfer;
  mg_transfer.build_matrices(dof_handler);

  //mg_transfer.print_matrices(std::cout);

  

  
  
}

  
} //end namespace stokes

#endif
