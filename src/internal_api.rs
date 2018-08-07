use std::cell::RefCell;
use std::collections::HashMap;

use function::Function;

// thread_local! {
//     pub(crate) static API: RefCell<HashMap<String, Function<'static>>> = RefCell::new(HashMap::new());
// }

// pub(crate) fn get(name: String) -> Option<Function<'static>> {
//     API.with(|hm| hm.borrow().get(&name).map(|f| f.clone()))
// }

// pub(crate) fn set(name: String, func: Function<'static>) {
//     API.with(|hm| {
//         (*hm.borrow_mut()).insert(name, func);
//     })
// }

// pub(crate) fn get_api(name: String, is_global: bool) -> Function<'static> {
//     let mut func = get(name.clone());
//     if func.is_none() {
//         func = Function::get_function(format!("{}", name), is_global, false);
//         set(name, func.clone().unwrap());
//     }
//     func.unwrap()
// }
