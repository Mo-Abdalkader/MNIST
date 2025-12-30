document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadArea = document.getElementById('upload-area');
    const resultArea = document.getElementById('result-area');
    const tryAnotherBtn = document.getElementById('try-another-btn');
    const featureMapsBtn = document.getElementById('feature-maps-btn');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictedDigit = document.getElementById('predicted-digit');
    const confidence = document.getElementById('confidence');
    const loading = document.getElementById('loading');

    // Event listeners
    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload();
        }
    });

    fileInput.addEventListener('change', handleFileUpload);

    tryAnotherBtn.addEventListener('click', function() {
        // Reset the application state
        fetch('/reset')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    resultArea.classList.add('hidden');
                    uploadArea.classList.remove('hidden');
                    fileInput.value = '';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // Fallback in case of fetch error
                resultArea.classList.add('hidden');
                uploadArea.classList.remove('hidden');
                fileInput.value = '';
            });
    });

    // Feature Maps button event listener
    if (featureMapsBtn) {
        featureMapsBtn.addEventListener('click', function() {
            window.location.href = '/feature_maps';
        });
    }

    // Handle file upload and prediction
    function handleFileUpload() {
        if (fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        if (!file.type.match('image.*')) {
            alert('Please upload an image file');
            return;
        }

        // Show loading state
        loading.classList.remove('hidden');

        // Create form data and send to server
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update UI with prediction results
            uploadedImage.src = data.image_path + '?t=' + new Date().getTime(); // Add timestamp to prevent caching
            predictedDigit.textContent = data.predicted_digit;
            confidence.textContent = data.confidence.toFixed(2) + '%';

            // Show results
            uploadArea.classList.add('hidden');
            resultArea.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please try again.');
        })
        .finally(() => {
            // Hide loading state
            loading.classList.add('hidden');
        });
    }
});