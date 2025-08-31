$(document).ready( function () {
    $('table.datatable').DataTable({
        paging: false,
        scrollCollapse: true,
        scrollY: '800px',
        order: [[4,'asc'],[0,'asc']],
    }).columns.align();
});
