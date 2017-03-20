#ifndef STOKES_PROBLEM_H
#define STOKES_PROBLEM_H

#include "globals.hpp"

#include <map>

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix_ez.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/matrix_block.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/grid/tria.h>


namespace stokes {

template<int dim>
class StokesProblem
{
public:
  StokesProblem(const unsigned int degree);
  ~StokesProblem() = default;
  
  void
  run ();

private:
  void make_grid_and_dofs();
  void assign_boundary_ids();
  void distill_matrices();
  void assemble_system_PreProcess();
  void assemble_system();
  void print_matrices();
  void assemble_system_PostProcess();
  void solve();
  void estimate_error();
  void output_results() const;

  const unsigned int degree;
  dealii::Tensor<1,dim> referenceDirection;
  dealii::Triangulation<dim> triangulation;
  dealii::FESystem<dim> fe;
  dealii::DoFHandler<dim> dof_handler;

  dealii::BlockSparsityPattern sparsity_pattern;
  dealii::BlockSparseMatrix<stokes::real> system_matrix;

  dealii::SparsityPattern
    massSigma_SP
    ,inverseMassSigma_SP
    ,stiffSigmaFromU_SP
    ,stiffUFromSigma_SP
    ,stiffUFromP_SP
    ,stiffPFromU_SP;

  dealii::SparseMatrix<stokes::real>
    massSigma
    ,inverseMassSigma
    ,stiffSigmaFromU
    ,stiffUFromSigma
    ,stiffUFromP
    ,stiffPFromU;

  dealii::BlockVector<stokes::real>
    solution
    , system_rhs;

  const dealii::MappingQ<dim,dim> mapping;

  // dealii::MGLevelObjects<dealii::SparsityPattern> mg_sparsity_patterns;
  // dealii::MGLevelObjects<dealii::SparsityPattern<stokes::real> >
  //   mg_matrices;
  

  dealii::MatrixBlockVector<dealii::SparseMatrix<stokes::real> > SystemMatricesCollection;
  dealii::AnyData SystemRHSsAndConstraints;

  std::map<dealii::types::boundary_id,
	   stokes::myBoundaryID> BoundaryIDMap;
};

  


} //end namespace stokes

#include "StokesProblem.tpp"

#endif
