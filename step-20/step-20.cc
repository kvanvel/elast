#include "globals.hpp"

#include "DirBC.hpp"
#include "NueBC.hpp"
#include "elasRHS.hpp"
#include "compliance.hpp"
#include "LDGIntegrator.hpp"
#include "LDGErrorIntegrator.hpp"


// Things that are obviously wrong:
// Need to put printing in a the correct spot.
// 

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

namespace Step20
{
using namespace dealii;


template <int dim>
class elasMixedLaplaceProblem
{
public:
  elasMixedLaplaceProblem (const unsigned int degree);
  void run ();

private:
  void make_grid_and_dofs ();
  void assign_boundary_ids();
  void distill_matrices();
  void assemble_system_PreProcess();    
  void assemble_system();
  void assemble_system_PostProcess();
  void solve_NEW();
  void EstimateError();
  //void compute_errors () const;
  void output_results () const;

  const unsigned int   degree;

  dealii::Point<dim> referenceDirection;

  dealii::Triangulation<dim>   triangulation;
  dealii::FESystem<dim>        fe;
  dealii::DoFHandler<dim>      dof_handler;

  dealii::BlockSparsityPattern      sparsity_pattern;
  dealii::BlockSparseMatrix<elas::real> system_matrix;
  

  dealii::SparsityPattern
    massSigma_SP,
    inverseMassSigma_SP,
    boundaryMassU_SP,
    boundaryMassQ_SP,
    StiffSigmaFromU_SP,
    StiffUFromSigma_SP,
    BoundaryMassUForDirMinus_SP,
    BoundaryMassQForNuePlus_SP,
    SvdU_ForDirMinus_SP,
    SvdU_ForNuePlus_SP,
    Svd_CapitalSigma_ForNuePlus_SP,
    Svd_CapitalSigma_ForDirMinus_SP,
    SvdV_ForNuePlus_SP,
    SvdV_ForDirMinus_SP;
  
  
    // BoundaryMassUForDirMinus_SP
    // BoundaryMassQForNuePlus_SP;

  dealii::SparseMatrix< elas::real >
  massSigma,
    Svd_U_for_Nue,
    Svd_Sigma_for_Nue,
    BoundaryMassSigma,
    InverseMassSigma,
    StiffSigmaFromU,
    MassU,
    Svd_U_for_Dir,
    Svd_Sigma_for_Dir,
    InverseMassU,
    BoundaryMassU,
    StiffUFromSigma;

  //TODO:  These need better names;
  dealii::SparseMatrixEZ<elas::real>
  PP,
    QQ,
    RR,
    SS;

  unsigned int 
  rankBoundaryMassU,
    nullityBoundaryMassU,
    rankBoundaryMassSigma,
    nullityBoundaryMassSigma;

  dealii::BlockVector<elas::real>       solution;
  dealii::BlockVector<elas::real>       system_rhs;
  dealii::BlockVector<elas::real>
  blockConstraints1,
    blockConstraints2;

  dealii::BlockVector<elas::real>  EstimatedError;
    

  const dealii::MappingQ<dim,dim> mapping;
  const dealii::MappingQ<dim-1,dim> faceMapping;

  dealii::MatrixBlockVector<dealii::SparseMatrix<elas::real> > SystemMatricesCollection;
  dealii::AnyData SystemRHSsAndConstraints;  

  std::map<dealii::types::boundary_id,
           elas::myBoundaryID> BoundaryIDMap;
  
};





template <int dim>
elasMixedLaplaceProblem<dim>::elasMixedLaplaceProblem (const unsigned int degree)
  :
  degree (degree),
  fe (dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree), (dim*dim+dim)/2),1,
      dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree), dim), 1 ),
  dof_handler (triangulation),  
  EstimatedError(2),
  mapping(1),
  faceMapping(1)  
{
  for(unsigned int i = 0; i < dim; ++i){
    referenceDirection[i] = 1.0;
  }

  referenceDirection[0] = 1.0;
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
     LDGErrorIntegrator(referenceDirection);

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
	  if(quad_point(0) <= 0.0 or true){
	    face_sys->set_boundary_id(dir_plus);
	  }
	  else {
	    face_sys->set_boundary_id(nue_plus);
	  }
	} else {
	  if(quad_point(0) <= 0.0 and false) {
	    face_sys->set_boundary_id(dir_minus);
	  } else {
	    face_sys->set_boundary_id(nue_minus);
	  }
	}	    
      }
    }
  }      
}

template <int dim>
void elasMixedLaplaceProblem<dim>::make_grid_and_dofs (){

  const dealii::types::boundary_id
    dir_plus = 1,
    dir_minus = 0,
    nue_plus = 3,
    nue_minus = 2;

  BoundaryIDMap[dir_plus ] = elas::myBoundaryID::dir_Plus;
  BoundaryIDMap[dir_minus ] = elas::myBoundaryID::dir_Minus;
  BoundaryIDMap[nue_plus ] = elas::myBoundaryID::nue_Plus;
  BoundaryIDMap[nue_minus ] = elas::myBoundaryID::nue_Minus;

  dealii::GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (3);

  dof_handler.distribute_dofs (fe);
  dof_handler.initialize_local_block_info();  
  
  dealii::DoFRenumbering::downstream(dof_handler, referenceDirection,false);
  dealii::DoFRenumbering::block_wise (dof_handler);

  std::vector<dealii::types::global_dof_index> dofs_per_component ((dim*dim+dim)/2+dim);

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
  
  {
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

    //sigma and u
    const unsigned int offset = (dim * dim + dim)/2;

    //u and u
    for(int i = 0; i < dim; ++i){
      for(int j = 0; j < dim; ++j){
	interiorMask[i+offset][j+offset] = dealii::DoFTools::nonzero;
      }
    }

    //u and sigma
    for(int i = 0; i < (dim * dim + dim )/2; ++i){
      for(int j = 0; j < dim; ++j){
	interiorMask[i][j+offset] = dealii::DoFTools::nonzero;
	fluxMask[i][j+offset] = dealii::DoFTools::nonzero;
	interiorMask[j+offset][i] = dealii::DoFTools::nonzero;
	fluxMask[j+offset][i] = dealii::DoFTools::nonzero;
      }
    }

    // for(unsigned int c = 0; c< fe.n_components(); ++c){
    //   for(unsigned int d = 0; d < fe.n_components(); ++d){
    // 	interiorMask[c][d]= dealii::DoFTools::none;
    // 	fluxMask[c][d]= dealii::DoFTools::none;
    //   }
    // }

     
    dealii::BlockDynamicSparsityPattern dsp(2,2);
    dsp.block(0,0).reinit(n_sigma,n_sigma);
    dsp.block(0,1).reinit(n_sigma,n_u);
    dsp.block(1,0).reinit(n_u,n_sigma);
    dsp.block(1,1).reinit(n_u,n_u);
    
    dsp.collect_sizes();
    
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler,dsp, interiorMask, fluxMask);
    
    sparsity_pattern.copy_from(dsp);        
    
  }


  const unsigned int
    n_couplings = 0;

  
    
  assert(fe.n_components() == (dim*dim+dim)/2 + dim );
    
  

  sparsity_pattern.compress();

  system_matrix.reinit (sparsity_pattern);

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
  SystemMatricesCollection.add(0,0,"MassSigma"); // 0
  SystemMatricesCollection.add(0,1,"StiffSigmaFromU"); //1
  SystemMatricesCollection.add(1,0,"StiffUFromSigma"); //2
  SystemMatricesCollection.add(1,1,"BoundaryMassForDirMinus");  //3
  SystemMatricesCollection.add(0,0,"BoundaryMassForNuePlus");  //4
  SystemMatricesCollection.add(0,0,"InverseMassSigma");        //5
  SystemMatricesCollection.add(1,1,"SvdU_ForDirMinus");        //6
  SystemMatricesCollection.add(1,1,"Svd_CapitalSigma_ForDirMinus");  //7
  SystemMatricesCollection.add(0,0,"SvdU_ForNuePlus"); //8
  SystemMatricesCollection.add(0,0,"Svd_CapitalSigma_ForNuePlus"); //9
  SystemMatricesCollection.add(0,0,"SvdV_ForNuePlus"); //10
  SystemMatricesCollection.add(1,1,"SvdV_ForDirMinus"); //11

  
  SystemMatricesCollection.reinit(sparsity_pattern);

  SystemRHSsAndConstraints.add(&system_rhs, "system_rhs");
  SystemRHSsAndConstraints.add(&blockConstraints1, "BlockConstraints1");

    
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
void elasMixedLaplaceProblem<dim>::assemble_system()
{
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


// template <int dim>
// void elasMixedLaplaceProblem<dim>::assemble_system_OLD ()
// {
//   QGauss<dim>   quadrature_formula(degree+2);
//   QGauss<dim-1> face_quadrature_formula(degree+2);
  
//   FEValues<dim> fe_values (fe, quadrature_formula,
// 			   update_values    | update_gradients |
// 			   update_quadrature_points  | update_JxW_values);


    
//   FEFaceValues<dim>
//     fe_face_values (fe, face_quadrature_formula,
// 		    update_values |
// 		    update_normal_vectors |
// 		    update_quadrature_points |
// 		    update_JxW_values);

//   FESubfaceValues<dim>
//     fe_subface_values (fe, face_quadrature_formula,
// 		       update_values |
// 		       update_normal_vectors |
// 		       update_quadrature_points
// 		       | update_JxW_values);

//   FEFaceValues<dim>
//     fe_face_values_neighbor (fe,
// 			     face_quadrature_formula,
// 			     update_values |
// 			     update_normal_vectors |
// 			     update_quadrature_points |
// 			     update_JxW_values);


		 
//   const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
//   const unsigned int   n_q_points      = quadrature_formula.size();
//   const unsigned int   n_face_q_points = face_quadrature_formula.size();


//   FullMatrix<double> vi_ui_matrix(dofs_per_cell,dofs_per_cell);
//   FullMatrix<double> vi_ue_matrix(dofs_per_cell,dofs_per_cell);
//   //FullMatrix<double> ve_ui_matrix(dofs_per_cell,dofs_per_cell);
//   //FullMatrix<double> ve_ue_matrix(dofs_per_cell,dofs_per_cell);
    
//   Vector<double>       local_rhs (dofs_per_cell);
  
//   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
//   std::vector<types::global_dof_index> neighbor_dof_indices (dofs_per_cell);

//   const elas::elasRHS<dim>         RHS;

//   const elas::Compliance<dim>  compliance;

//   const elas::DirBC<dim> DirBC;
//   const elas::NueBC<dim> NueBC;

//   std::vector<SymmetricTensor<4,dim> > compliance_values(n_q_points);
//   std::vector<Tensor<1,dim> > rhs_values;

//   const FEValuesExtractors::SymmetricTensor<2> stress (0);
//   const FEValuesExtractors::Vector displacements ((dim*dim+dim)/2);

    
//   typename DoFHandler<dim>::active_cell_iterator
//     cell = dof_handler.begin_active(),
//     endc = dof_handler.end();
//   for (; cell!=endc; ++cell)
//     {
//       fe_values.reinit (cell);
//       vi_ui_matrix = 0;
//       local_rhs = 0;	

//       compliance.value_list (fe_values.get_quadrature_points(),
// 			     compliance_values);
	

//       for (unsigned int q=0; q<n_q_points; ++q)
// 	for (unsigned int i=0; i<dofs_per_cell; ++i){
	    
// 	  const SymmetricTensor<2,dim> phi_i_tau
// 	    =  fe_values[stress].value(i,q);
	    
// 	  const Tensor<1,dim> div_phi_i_tau
// 	    = fe_values[stress].divergence(i,q);
	    
// 	  const Tensor<1,dim> phi_i_v
// 	    = fe_values[displacements].value (i, q);

// 	  const SymmetricTensor<2,dim> symGrad_phi_i_v
// 	    = fe_values[displacements].symmetric_gradient(i,q);
		
// 	  for (unsigned int j=0; j<dofs_per_cell; ++j){
// 	    const SymmetricTensor<2,dim> phi_j_sigma
// 	      = fe_values[stress].value (j, q);
// 	    const Tensor<1,dim>  phi_j_u
// 	      = fe_values[displacements].value (j, q);
	      
// 	    vi_ui_matrix(i,j)
// 	      +=
// 	      phi_i_tau 
// 	      * compliance_values[q]
// 	      * phi_j_sigma
// 	      * fe_values.JxW(q)
// 	      - div_phi_i_tau * phi_j_u * fe_values.JxW(q)
// 	      - symGrad_phi_i_v * phi_j_sigma* fe_values.JxW(q);

// 	  }
// 	  local_rhs(i) +=
// 	    phi_i_v * RHS.value(fe_values.quadrature_point(q))*	    
// 	    fe_values.JxW(q);
// 	}
	
//       for (unsigned int face_no=0;
// 	   face_no < GeometryInfo<dim>::faces_per_cell;
// 	   ++face_no){
// 	typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
// 	if (face->at_boundary() ){	  
	    
// 	  fe_face_values.reinit(cell,face_no);
// 	  const std::vector<double> & JxW = fe_face_values.get_JxW_values ();
// 	  const std::vector<Point<dim> >  & normals = fe_face_values.get_normal_vectors();	    
	    
	    
// 	  for(unsigned int point = 0; point< fe_face_values.n_quadrature_points; ++point){
// 	    const double refDotNormal = referenceDirection * normals[point];
// 	    if( refDotNormal > 0){
// 	      for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		for(unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j){	
// 		  vi_ui_matrix(i,j) +=
// 		    fe_face_values[displacements].value(i,point)
// 		    * ( fe_face_values[stress].value(j,point) 
// 		    * normals[point]) * JxW[point];		      
// 		}

// 		local_rhs(i) += (fe_face_values[stress].value(i,point) * DirBC.value(fe_face_values.quadrature_point(point) ))*normals[point] * JxW[point]; 
// 	      }
// 	    }else{
// 	      for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		for(unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j){
// 		  vi_ui_matrix(i,j) +=
// 		    ( fe_face_values[stress].value(i,point)*
// 		      fe_face_values[displacements].value(j,point) ) * normals[point] * JxW[point];
// 		}		
// 		local_rhs(i) += (fe_face_values[displacements].value(i,point)* (NueBC.value(fe_face_values.quadrature_point(point))* normals[point] ) * JxW[point]);
// 	      }
// 	    }
		
// 	  }
	    
// 	} else { 
// 	  Assert(
// 		 cell->neighbor(face_no).state()
// 		 == IteratorState::valid,
// 		 ExcInternalError());

// 	  typename DoFHandler<dim>::cell_iterator neighbor =
// 	    cell->neighbor(face_no);
	    
// 	  assert(  ! face->has_children() );
	    
// 	  // if( !cell->neighbor_is_coarser(face_no) &&
// 	  // 	(neighbor->index() > cell->index() ||
// 	  // 	 (neighbor->level() < cell->level() &&
// 	  // 	  neighbor->index() == cell->index()))){
// 	  if(true){
// 	    const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);	      
		 
	      
// 	    vi_ue_matrix = 0;
// 	    //ve_ui_matrix = 0;
// 	    //ve_ue_matrix = 0;

// 	    fe_face_values.reinit(cell,face_no);
// 	    fe_face_values_neighbor.reinit(neighbor,neighbor2);
	      
// 	    const std::vector<double>      & JxW = fe_face_values.get_JxW_values ();
// 	    const std::vector<Point<dim> > & normals = fe_face_values.get_normal_vectors();

// 	    //std::vector<Point<dim> > beta (fe_face_values.n_quadrature_points);
// 	    // Point<dim> referenceDirection; 
// 	    // for (unsigned int iii = 0; iii < dim; ++iii){
// 	    //   referenceDirection[iii] = 1.0;
// 	    // }

	    
// 	    for(unsigned int point = 0; point< fe_face_values.n_quadrature_points; ++point){
// 	      const double refDotNormal = referenceDirection * normals[point];
// 	      if( refDotNormal > 0){
// 		for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		  for(unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j){	
// 		    vi_ui_matrix(i,j) +=
// 		      fe_face_values[displacements].value(i,point)
// 		      * (fe_face_values[stress].value(j,point) 
// 		      * normals[point] ) * JxW[point];
// 		  }
// 		}
// 		for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		  for(unsigned int j = 0; j < fe_face_values_neighbor.dofs_per_cell;++j){
// 		    vi_ue_matrix(i,j) +=
// 		      (fe_face_values[stress].value(i,point) *
// 		       fe_face_values_neighbor[displacements].value(j,point) ) * normals[point]  
// 		        * JxW[point];
// 		  }
// 		}

// 	      }else{
// 		for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		  for(unsigned int j = 0; j < fe_face_values_neighbor.dofs_per_cell; ++j){
// 		    vi_ue_matrix(i,j) +=
// 		      fe_face_values[displacements].value(i,point)
// 		      * (fe_face_values_neighbor[stress].value(j,point) 
// 			 * normals[point] ) * JxW[point];
// 		  }
// 		}
// 		for(unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i){
// 		  for(unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j){
// 		    vi_ui_matrix(i,j) +=
// 		      (fe_face_values[stress].value(i,point)*
// 		       fe_face_values[displacements].value(j,point)) *normals[point]  * JxW[point];
// 		  }
// 		}
// 	      }
// 	    }
// 	  }
// 	  //insert
// 	  neighbor->get_dof_indices(neighbor_dof_indices);
// 	  cell->get_dof_indices (local_dof_indices);

// 	  for(unsigned int i = 0; i < dofs_per_cell; ++i){
// 	    for(unsigned int j = 0; j < dofs_per_cell; ++j){		
// 	      system_matrix.add( local_dof_indices[i],
// 				 neighbor_dof_indices[j],
// 				 vi_ue_matrix(i,j) );
// 	    }
// 	  }	    
// 	}
//       } //end of loop through faces	

//       cell->get_dof_indices (local_dof_indices);	
//       for (unsigned int i=0; i<dofs_per_cell; ++i){
// 	for (unsigned int j=0; j<dofs_per_cell; ++j){
// 	  system_matrix.add (local_dof_indices[i],
// 			     local_dof_indices[j],
// 			     vi_ui_matrix(i,j));	      
// 	}	  
//       }
//       for (unsigned int i = 0; i < dofs_per_cell; ++i){
// 	system_rhs(local_dof_indices[i]) += local_rhs(i);
//       }
//     } //end of loop through cells
// }



// class SchurComplement : public Subscriptor
// {
// public:
//   SchurComplement (const BlockSparseMatrix<double> &A,
// 		   const IterativeInverse<Vector<double> > &Minv);

//   void vmult (Vector<double>       &dst,
// 	      const Vector<double> &src) const;

// private:
//   const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
//   const SmartPointer<const IterativeInverse<Vector<double> > > m_inverse;

//   mutable Vector<double> tmp1, tmp2;
// };


// SchurComplement::SchurComplement (const BlockSparseMatrix<double> &A,
// 				  const IterativeInverse<Vector<double> > &Minv)
//   :
//   system_matrix (&A),
//   m_inverse (&Minv),
//   tmp1 (A.block(0,0).m()),
//   tmp2 (A.block(0,0).m())
// {}


// void SchurComplement::vmult (Vector<double>       &dst,
// 			     const Vector<double> &src) const
// {
//   system_matrix->block(0,1).vmult (tmp1, src);
//   m_inverse->vmult (tmp2, tmp1);
//   system_matrix->block(1,0).vmult (dst, tmp2);
// }


// class ApproximateSchurComplement : public Subscriptor
// {
// public:
//   ApproximateSchurComplement (const BlockSparseMatrix<double> &A);

//   void vmult (Vector<double>       &dst,
// 	      const Vector<double> &src) const;
//   void Tvmult (Vector<double>       &dst,
// 	       const Vector<double> &src) const;

// private:
//   const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

//   mutable Vector<double> tmp1, tmp2;
// };


// ApproximateSchurComplement::ApproximateSchurComplement (const BlockSparseMatrix<double> &A)
//   :
//   system_matrix (&A),
//   tmp1 (A.block(0,0).m()),
//   tmp2 (A.block(0,0).m())
// {}


// void ApproximateSchurComplement::vmult (Vector<double>       &dst,
// 					const Vector<double> &src) const
// {
//   system_matrix->block(0,1).vmult (tmp1, src);
//   system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
//   system_matrix->block(1,0).vmult (dst, tmp2);
// }


// void ApproximateSchurComplement::Tvmult (Vector<double>       &dst,
// 					 const Vector<double> &src) const
// {
//   vmult (dst, src);
// }



// template <int dim>
// void MixedLaplaceProblem<dim>::solve ()
// {
//   PreconditionIdentity identity;
//   IterativeInverse<Vector<double> > m_inverse;
//   m_inverse.initialize(system_matrix.block(0,0), identity);
//   m_inverse.solver.select("cg");
//   static ReductionControl inner_control(1000, 0., 1.e-13);
//   m_inverse.solver.set_control(inner_control);

//   Vector<double> tmp (solution.block(0).size());

//   {
//     Vector<double> schur_rhs (solution.block(1).size());

//     m_inverse.vmult (tmp, system_rhs.block(0));
//     system_matrix.block(1,0).vmult (schur_rhs, tmp);
//     schur_rhs -= system_rhs.block(1);


//     SchurComplement
//       schur_complement (system_matrix, m_inverse);

//     ApproximateSchurComplement
//       approximate_schur_complement (system_matrix);

//     IterativeInverse<Vector<double> >
//       preconditioner;
//     preconditioner.initialize(approximate_schur_complement, identity);
//     preconditioner.solver.select("cg");
//     preconditioner.solver.set_control(inner_control);


//     SolverControl solver_control (solution.block(1).size(),
// 				  1e-12*schur_rhs.l2_norm());
//     SolverCG<>    cg (solver_control);

//     cg.solve (schur_complement, solution.block(1), schur_rhs,
// 	      preconditioner);

//     std::cout << solver_control.last_step()
// 	      << " CG Schur complement iterations to obtain convergence."
// 	      << std::endl;
//   }

//   {
//     system_matrix.block(0,1).vmult (tmp, solution.block(1));
//     tmp *= -1;
//     tmp += system_rhs.block(0);

//     m_inverse.vmult (solution.block(0), tmp);
//   }
// }

// // template <int dim>
// // void elasMixedLaplaceProblem<dim>::solve ()
// // {


// //   PreconditionIdentity identity;
// //   IterativeInverse<Vector<double> > m_inverse;
// //   m_inverse.initialize(system_matrix.block(0,0), identity);
// //   m_inverse.solver.select("cg");
// //   static ReductionControl inner_control(1000, 0., 1.e-13);
// //   m_inverse.solver.set_control(inner_control);

//   Vector<double> tmp (solution.block(0).size());

//   {
//     Vector<double> schur_rhs (solution.block(1).size());

//     m_inverse.vmult (tmp, system_rhs.block(0));
//     system_matrix.block(1,0).vmult (schur_rhs, tmp);
//     schur_rhs -= system_rhs.block(1);


//     SchurComplement
//       schur_complement (system_matrix, m_inverse);

//     ApproximateSchurComplement
//       approximate_schur_complement (system_matrix);

//     IterativeInverse<Vector<double> >
//       preconditioner;
//     preconditioner.initialize(approximate_schur_complement, identity);
//     preconditioner.solver.select("cg");
//     preconditioner.solver.set_control(inner_control);


//     SolverControl solver_control (solution.block(1).size(),
// 				  1e-12*schur_rhs.l2_norm());
//     SolverCG<>    cg (solver_control);

//     cg.solve (schur_complement, solution.block(1), schur_rhs,
// 	      preconditioner);

//     std::cout << solver_control.last_step()
// 	      << " CG Schur complement iterations to obtain convergence."
// 	      << std::endl;
//   }

//   {
//     system_matrix.block(0,1).vmult (tmp, solution.block(1));
//     tmp *= -1;
//     tmp += system_rhs.block(0);

//     m_inverse.vmult (solution.block(0), tmp);
//   }
// }

template <int dim>
void elasMixedLaplaceProblem<dim>::solve_NEW ()
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
  
  
  //

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
    PPSigmaPPStar
    = dealii::linear_operator(PP)
    * dealii::linear_operator(*svdSigmaForNue)
    * dealii::transpose_operator( dealii::linear_operator(PP ));

  auto Id = dealii::PreconditionIdentity();
  dealii::SolverControl firstSolverControl(100,1e-14);
  dealii::SolverCG<dealii::Vector<elas::real> > firstSolver(firstSolverControl);

  const auto PPSigmaPPStar_Inverse
    = dealii::inverse_operator(PPSigmaPPStar,
			       firstSolver,
			       Id);

  
    
  dealii::Vector<elas::real>  scriptm;
  //find script lower case m;  
  if(rankBoundaryMassSigma > 0){    
    // dealii::Vector<elas::real> fakeNueConstraintsRHS;
    
    
    // fakeNueConstraintsRHS.reinit(svdSigmaForNue->m() );
    // for(unsigned int i = 0; i < svdSigmaForNue->m();++i){
    //   fakeNueConstraintsRHS(i) = 0;
    // }


    scriptm.reinit(rankBoundaryMassSigma);
    dealii::Vector<elas::real> localRHS( PP.n() );

    dealii::transpose_operator(scriptP).vmult
      (localRHS,
       SystemRHSsAndConstraints.entry<dealii::BlockVector<elas::real>* >(1)->block(0));
 
    PPSigmaPPStar_Inverse.vmult(scriptm,localRHS);

    scriptP.vmult(solution.block(0), scriptm);
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
    dealii::Vector<elas::real> fakeDirConstraintsRHS;
    
    fakeDirConstraintsRHS.reinit(svdSigmaForDir->m() );
    for(unsigned int i = 0; i < svdSigmaForDir->m(); ++i){
      fakeDirConstraintsRHS(i) = 0.0;
    }
    scriptl.reinit(rankBoundaryMassU);
    
    //dealii::Vector<elas::real> localRHS(scriptR.n() );
    dealii::Vector<elas::real> localRHS(RR.n() );
    (dealii::transpose_operator(scriptR)).vmult
      (localRHS,
       SystemRHSsAndConstraints.entry<dealii::BlockVector<elas::real> * >(1)->block(1));
       //fakeDirConstraintsRHS);

    RRSigmaRRStar_Inverse.vmult(scriptl, localRHS);

    scriptR.vmult(solution.block(1), scriptl);
  }

  
  
  //make double struck F

  //dealii::Vector<elas::real> doubleStruckF (scriptQ.n() );
  dealii::Vector<elas::real> doubleStruckF (QQ.n() );
  
  dealii::transpose_operator(scriptQ).vmult(
					    doubleStruckF,
					    SystemRHSsAndConstraints.
					    entry<dealii::BlockVector<elas::real>* >(0)->block(0));

  if(rankBoundaryMassSigma > 0){
    const auto product1 =
      dealii::transpose_operator(scriptQ)
      * dealii::linear_operator(*MassSigma)
      * scriptP;
    product1.vmult_add(doubleStruckF, scriptm);
    

    const auto product2
      = dealii::transpose_operator(scriptQ)
      * dealii::linear_operator(*StiffSigmaFromU)
      * scriptR;
    product2.vmult_add(doubleStruckF, scriptl);
  }

  PRINT( "DoubleStruckG");
  //make double struck G
  dealii::Vector<elas::real> doubleStruckG (SS.n() ) ;
  dealii::transpose_operator(scriptS).vmult(
					    doubleStruckG,
					    SystemRHSsAndConstraints.
					    entry<dealii::BlockVector<elas::real>* >(0)->block(1));
  
  
  if(rankBoundaryMassU > 0){
    const auto product
      = dealii::transpose_operator(scriptS)
      * dealii::linear_operator(*StiffUFromSigma )
      * scriptP;

    product.vmult_add(doubleStruckG, scriptm);
  }  

  
  PRINT( "DoubleStruckG");
  //make script capital A;
  const auto
    QstarMQ =
    dealii::transpose_operator(scriptQ)
    * dealii::linear_operator(*MassSigma)
    * scriptQ;
  std::cout << __LINE__ << std::endl;
  const auto
  QstarMQ_ApproxInverse =
    dealii::transpose_operator(scriptQ)
    * dealii::linear_operator(*InverseMassSigma)
    * scriptQ;
std::cout << __LINE__ << std::endl;
  dealii::SolverControl InnerControlForQstarMQ(1000, 1e-14);
  dealii::SolverCG<> innerSolver(InnerControlForQstarMQ);
std::cout << __LINE__ << std::endl;
  const auto
    QstarMQ_Inverse
    = dealii::inverse_operator(QstarMQ, innerSolver, QstarMQ_ApproxInverse);
  std::cout << __LINE__ << std::endl;

  const auto
  scriptB
    = dealii::transpose_operator( dealii::linear_operator(scriptQ))
    * dealii::linear_operator(*StiffSigmaFromU)
    * scriptS;

  const auto
    scriptA =
    dealii::transpose_operator(scriptB)    
    * QstarMQ_Inverse
    * scriptB;
  
  // const auto
  //   scriptA
  //   = dealii::transpose_operator(scriptS)
  //   * dealii::linear_operator(*StiffUFromSigma)
  //   * scriptQ
  //   * QstarMQ_Inverse
  //   * dealii::transpose_operator( dealii::linear_operator(scriptQ))
  //   * dealii::linear_operator(*StiffSigmaFromU)
  //   * scriptS;
  std::cout << __LINE__ << std::endl;
  // const auto
  //   scriptA_Approx    
  //   = dealii::transpose_operator(scriptS)
  //   * dealii::linear_operator(*StiffUFromSigma)
  //   * scriptQ
  //   * QstarMQ_ApproxInverse
  //   * dealii::transpose_operator( dealii::linear_operator(scriptQ))
  //   * dealii::linear_operator(*StiffSigmaFromU)
  //   * scriptS;

  const auto
    scriptA_Approx
    = dealii::transpose_operator(scriptB)
    * QstarMQ_ApproxInverse
    * scriptB;
  
  dealii::SolverControl InnerControlForPrecon(1000, 1e-12);
  dealii::SolverCG<> PreconSolver(InnerControlForPrecon);
  
  const auto ID = dealii::PreconditionIdentity();
  //A pretty good approximation of the inverse is the inverse of the approximation.
  const auto scriptA_inverse_Approx = dealii::inverse_operator(scriptA_Approx, PreconSolver, ID);
  
  dealii::SolverControl outerControl(1000, 1e-10);
  
  dealii::SolverCG<> outerSolver(outerControl);  

  const auto scriptA_inverse = dealii::inverse_operator(scriptA, outerSolver,
							ID);
    //scriptA_inverse_Approx);

  dealii::Vector<elas::real> alpha (SS.n() );


  std::cout << "Solving for alpha" << std::endl;  
  scriptA_inverse.vmult(alpha, doubleStruckG);
  std::cout << "End Solving for alpha" << std::endl;

  scriptS.vmult_add( solution.block(1), alpha);

  //compute beta

  scriptB.vmult_add( doubleStruckF,alpha);

  dealii::Vector<elas::real> beta (QQ.n() );

  QstarMQ_Inverse.vmult(beta,doubleStruckF);

  scriptQ.vmult_add(solution.block(0), beta);


}


// template <int dim>
// void MixedLaplaceProblem<dim>::compute_errors () const
// {
//   const ComponentSelectFunction<dim>
//     pressure_mask (dim, dim+1);
//   const ComponentSelectFunction<dim>
//     velocity_mask(std::make_pair(0, dim), dim+1);

//   ExactSolution<dim> exact_solution;
//   Vector<double> cellwise_errors (triangulation.n_active_cells());

//   QTrapez<1>     q_trapez;
//   QIterated<dim> quadrature (q_trapez, degree+2);

//   VectorTools::integrate_difference (dof_handler, solution, exact_solution,
// 				     cellwise_errors, quadrature,
// 				     VectorTools::L2_norm,
// 				     &pressure_mask);
//   const double p_l2_error = cellwise_errors.l2_norm();

//   VectorTools::integrate_difference (dof_handler, solution, exact_solution,
// 				     cellwise_errors, quadrature,
// 				     VectorTools::L2_norm,
// 				     &velocity_mask);
//   const double u_l2_error = cellwise_errors.l2_norm();

//   std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
// 	    << ",   ||e_u||_L2 = " << u_l2_error
// 	    << std::endl;
// }


// template <int dim>
// void MixedLaplaceProblem<dim>::output_results () const
// {
//   std::vector<std::string> solution_names;
//   switch (dim)
//     {
//     case 2:
//       solution_names.push_back ("u");
//       solution_names.push_back ("v");
//       solution_names.push_back ("p");
//       break;

//     case 3:
//       solution_names.push_back ("u");
//       solution_names.push_back ("v");
//       solution_names.push_back ("w");
//       solution_names.push_back ("p");
//       break;

//     default:
//       Assert (false, ExcNotImplemented());
//     }


//   DataOut<dim> data_out;

//   data_out.attach_dof_handler (dof_handler);
//   data_out.add_data_vector (solution, solution_names);

//   data_out.build_patches (degree+1);

//   std::ofstream output ("solution.gmv");
//   data_out.write_gmv (output);
// }

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
  make_grid_and_dofs();
  assign_boundary_ids();
  assemble_system_PreProcess();
  assemble_system();
  assemble_system_PostProcess();
  
  //assemble_system_OLD ();
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
      
      name += ".mtx";
      std::ofstream ostr(name);
      PrintMatrixMarket(SystemMatricesCollection.matrix(i), ostr);      
    }    
    
  }

  solve_NEW ();
  output_results ();
  EstimateError();
  
}

}


int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      deallog.depth_console (0);

      elasMixedLaplaceProblem<2> mixed_laplace_problem(2);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
