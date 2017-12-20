
function genmodel(modalid, imageid){
    // Get the modal
    var modal = document.getElementById(modalid);

    // Get the image and insert it inside the modal - use its "alt" text as a caption
    var img = document.getElementById(imageid);
    img.onclick = function(){modal.style.display = "block";}

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];

    span.onclick = function() { modal.style.display = "none";}

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
    if (event.target == modal) {modal.style.display = "none";}
    }
}

genmodel('Modal1','Image1');