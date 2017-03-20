#ifndef __ELAS_MIXED_LAPLACE_PROBLEM_H
#define __ELAS_MIXED_LAPLACE_PROBLEM_H

//#include "globals.hpp"
#include "LDGIntegrator.hpp"
#include "LDGErrorIntegrator.hpp"
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_ez.h>

#include "globals.hpp"
#include <map>






namespace elas
{
using namespace dealii;


template <int dim>
class elasMixedLaplaceProblem
{
public:
  elasMixedLaplaceProblem (const unsigned int degree);
  void run ();

private:
  void refine_grid();
  void make_grid();
  void setup_system();
  //void make_grid_and_dofs ();
  void assign_boundary_ids();
  void distill_matrices();
  void assemble_system_PreProcess();    
  void assemble_system();
  void assemble_system_mg();
  void assemble_system_PostProcess();
  
  void solve();
  void EstimateError();  
  void output_results () const;

  const unsigned int   degree;

  dealii::Tensor<1,dim> referenceDirection;

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

  dealii::MGMatrixBlockVector<dealii::SparseMatrix<elas::real> >
    mg_SystemMatricesCollection;

  dealii::MGLevelObject<SparseMatrix<double> > mg_matrices;
  // dealii::MGLevelObject<dealii::MatrixBlockVector<dealii::SparseMatrix<elas::real> > >
  // mg_SystemMatricesCollection;
  
  dealii::MGLevelObject<dealii::BlockSparsityPattern>      mg_sparsity_patterns;
  dealii::MGLevelObject<SparseMatrix<elas::real> > mg_interface_in;
  dealii::MGLevelObject<SparseMatrix<elas::real> > mg_interface_out;

  std::map<dealii::types::boundary_id,
           elas::myBoundaryID> BoundaryIDMap;
  
};

}



#endif
