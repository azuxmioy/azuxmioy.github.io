// JS for demo
$( document ).ready(function() {
$( "#dialog1" ).dialog({
			autoOpen: false,
			show: {
				effect: "fade",
				duration: 150
			},
			hide: {
				effect: "fade",
				duration: 150
			},
			position: {
  			my: "center",
  			at: "center"
			},
			// Add the 2 options below to use click outside feature
			clickOutside: true, // clicking outside the dialog will close it
			clickOutsideTrigger: "#opener1"  // Element (id or class) that triggers the dialog opening 
		});
		
    $( "#dialog2" ).dialog({
			autoOpen: false,
			show: {
				effect: "fade",
				duration: 150
			},
			hide: {
				effect: "fade",
				duration: 150
			},
			position: {
  			my: "center",
  			at: "center"
			},
			// Add the 2 options below to use click outside feature
			clickOutside: true, // clicking outside the dialog will close it
			clickOutsideTrigger: "#opener2"  // Element (id or class) that triggers the dialog opening
		});		

    $( "#dialog3" ).dialog({
			autoOpen: false,
			show: {
				effect: "fade",
				duration: 150
			},
			hide: {
				effect: "fade",
				duration: 150
			},
			position: {
  			my: "center",
  			at: "center"
			},
			clickOutside: false // For demo purpose. Not necessary because this is the default value
		});	

		$( "#opener1" ).click(function() {
			$( "#dialog1" ).dialog( "open" );
		});
		
		$( "#opener2" ).click(function() {
			$( "#dialog2" ).dialog( "open" );
		});
		
		$( "#opener3" ).click(function() {
			$( "#dialog3" ).dialog( "open" );
		});
});

/* jQuery UI dialog clickoutside */

/*
The MIT License (MIT)

Copyright (c) 2013 - AGENCE WEB COHERACTIO

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

$.widget( "ui.dialog", $.ui.dialog, {
  options: {
    clickOutside: false, // Determine if clicking outside the dialog shall close it
    clickOutsideTrigger: "" // Element (id or class) that triggers the dialog opening 
  },

  open: function() {
    var clickOutsideTriggerEl = $( this.options.clickOutsideTrigger );
    var that = this;
    
    if (this.options.clickOutside){
      // Add document wide click handler for the current dialog namespace
      $(document).on( "click.ui.dialogClickOutside" + that.eventNamespace, function(event){
        if ( $(event.target).closest($(clickOutsideTriggerEl)).length == 0 && $(event.target).closest($(that.uiDialog)).length == 0){
          that.close();
        }
      });
    }
    
    this._super(); // Invoke parent open method
  },
  
  close: function() {
    var that = this;
    
    // Remove document wide click handler for the current dialog
    $(document).off( "click.ui.dialogClickOutside" + that.eventNamespace );
    
    this._super(); // Invoke parent close method 
  },  

});