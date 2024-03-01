# -*- coding: utf-8 -*-


from brainpy._src.tools.codes import (
  repr_object as repr_object,
  repr_dict as repr_dict,
  repr_context as repr_context,
  copy_doc as copy_doc,
  code_lines_to_func as code_lines_to_func,
  get_identifiers as get_identifiers,
  indent as indent,
  deindent as deindent,
  word_replace as word_replace,
  is_lambda_function as is_lambda_function,
  get_main_code as get_main_code,
  get_func_source as get_func_source,
  change_func_name as change_func_name,
)

from brainpy._src.tools.dicts import (
  DotDict as DotDict,
)

from brainpy._src.tools.math_util import (
  format_seed as format_seed,
)

from brainpy._src.tools.package import (
  numba_jit as numba_jit,
  numba_seed as numba_seed,
  numba_range as numba_range,
)

from brainpy._src.tools.others import (
  replicate as replicate,
  not_customized as not_customized,
  to_size as to_size,
  size2num as size2num,
  timeout as timeout,
  init_progress_bar as init_progress_bar,
)

from brainpy._src.tools.install import (
  jaxlib_install_info,
)



