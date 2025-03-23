// DiabeTongue - Diabetes Prediction Through Tongue Analysis
// Main JavaScript File

document.addEventListener('DOMContentLoaded', function() {
    // Image upload and preview functionality
    const uploadInput = document.getElementById('tongue-image');
    const uploadDropzone = document.getElementById('upload-dropzone');
    const uploadPrompt = document.querySelector('.upload-prompt');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const changeImageBtn = document.getElementById('change-image');
    
    if (uploadInput) {
        // Handle file selection
        uploadInput.addEventListener('change', function(event) {
            handleFileSelect(event);
        });
        
        // Handle drag and drop
        uploadDropzone.addEventListener('dragover', function(event) {
            event.preventDefault();
            event.stopPropagation();
            uploadDropzone.classList.add('dragover');
        });
        
        uploadDropzone.addEventListener('dragleave', function(event) {
            event.preventDefault();
            event.stopPropagation();
            uploadDropzone.classList.remove('dragover');
        });
        
        uploadDropzone.addEventListener('drop', function(event) {
            event.preventDefault();
            event.stopPropagation();
            uploadDropzone.classList.remove('dragover');
            
            if (event.dataTransfer.files.length) {
                uploadInput.files = event.dataTransfer.files;
                handleFileSelect({ target: uploadInput });
            }
        });
        
        // Handle change image button
        if (changeImageBtn) {
            changeImageBtn.addEventListener('click', function() {
                // Reset the file input
                uploadInput.value = '';
                // Hide preview, show upload prompt
                imagePreviewContainer.style.display = 'none';
                uploadPrompt.style.display = 'block';
            });
        }
    }
    
    // Function to handle file selection
    function handleFileSelect(event) {
        const file = event.target.files[0];
        
        if (file) {
            // Check if the file is an image
            if (!file.type.match('image.*')) {
                alert('Please select an image file.');
                return;
            }
            
            // Create a FileReader to read the image
            const reader = new FileReader();
            
            reader.onload = function(e) {
                // Update the image preview
                imagePreview.src = e.target.result;
                // Show the preview container, hide the upload prompt
                imagePreviewContainer.style.display = 'block';
                uploadPrompt.style.display = 'none';
            };
            
            // Read the image file
            reader.readAsDataURL(file);
        }
    }
    
    // Form validation
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            // Check if a file is selected
            if (uploadInput.files.length === 0) {
                event.preventDefault();
                alert('Please select a tongue image for analysis.');
                return;
            }
        });
    }
});
