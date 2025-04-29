/**
 * Custom JavaScript for Maintenance Optimization Application
 */

// Wait for the document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Function to show notification messages
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.role = 'alert';
        
        // Add message content
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Find the container to append the notification
        const container = document.querySelector('.container');
        if (container) {
            // Insert at the top of the container
            container.insertBefore(notification, container.firstChild);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => {
                    notification.remove();
                }, 300); // Wait for fade animation
            }, 5000);
        }
    }
    
    // Function to confirm dangerous actions
    function confirmAction(message, callback) {
        if (confirm(message)) {
            callback();
        }
    }
    
    // Function to format numbers with commas
    function formatNumber(number) {
        return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
    
    // Function to handle error responses from AJAX calls
    function handleAjaxError(error) {
        console.error('Error:', error);
        showNotification('An error occurred. Please check the console for details.', 'danger');
    }
    
    // Function to enhance charts and visualizations
    function enhanceVisualizations() {
        // Find all images in visualization containers
        document.querySelectorAll('.visualization-container img').forEach(img => {
            // Add lightbox functionality
            img.addEventListener('click', function() {
                // Create lightbox
                const lightbox = document.createElement('div');
                lightbox.className = 'lightbox';
                lightbox.style.position = 'fixed';
                lightbox.style.top = '0';
                lightbox.style.left = '0';
                lightbox.style.width = '100%';
                lightbox.style.height = '100%';
                lightbox.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                lightbox.style.display = 'flex';
                lightbox.style.justifyContent = 'center';
                lightbox.style.alignItems = 'center';
                lightbox.style.zIndex = '9999';
                
                // Add lightbox content
                const imgClone = this.cloneNode();
                imgClone.style.maxWidth = '90%';
                imgClone.style.maxHeight = '90%';
                imgClone.style.objectFit = 'contain';
                imgClone.style.border = 'none';
                
                lightbox.appendChild(imgClone);
                
                // Add close on click
                lightbox.addEventListener('click', function() {
                    lightbox.remove();
                });
                
                // Add to body
                document.body.appendChild(lightbox);
            });
            
            // Make it clear this is clickable
            img.style.cursor = 'pointer';
            img.title = 'Click to enlarge';
        });
    }
    
    // Function to add export functionality for data
    function addExportFunctionality() {
        // Add export buttons to visualization container
        document.querySelectorAll('.visualization-container').forEach(container => {
            const exportButton = document.createElement('button');
            exportButton.className = 'btn btn-sm btn-outline-primary mt-2';
            exportButton.innerText = 'Export Image';
            
            exportButton.addEventListener('click', function() {
                const img = container.querySelector('img');
                if (img) {
                    const a = document.createElement('a');
                    a.href = img.src;
                    a.download = 'maintenance-optimization-chart.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            });
            
            container.appendChild(exportButton);
        });
    }
    
    // Add responsive behavior to tables
    function makeTablesResponsive() {
        document.querySelectorAll('table').forEach(table => {
            if (!table.classList.contains('table-responsive')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'table-responsive';
                table.parentNode.insertBefore(wrapper, table);
                wrapper.appendChild(table);
            }
        });
    }
    
    // Confirm reset simulation
    const resetButton = document.getElementById('reset-simulation');
    if (resetButton) {
        resetButton.addEventListener('click', function(e) {
            e.preventDefault();
            confirmAction('Are you sure you want to reset the simulation? This will clear all current results.', function() {
                // Allow the original click handler to proceed
                // This assumes the click handler is attached via jQuery
                $(resetButton).trigger('click.original');
            });
        });
    }
    
    // Format numbers in metrics display
    function formatMetricNumbers() {
        document.querySelectorAll('[id$="-cost"]').forEach(el => {
            const value = el.textContent.trim();
            if (value !== 'N/A' && !isNaN(parseFloat(value))) {
                el.textContent = formatNumber(parseFloat(value));
            }
        });
    }
    
    // Check if we're on the visualization page
    if (window.location.pathname.includes('/visualization')) {
        enhanceVisualizations();
        addExportFunctionality();
        makeTablesResponsive();
        formatMetricNumbers();
    }
    
    // Check if we're on the policy page
    if (window.location.pathname.includes('/policy')) {
        enhanceVisualizations();
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Expose utility functions globally
    window.maintenanceApp = {
        showNotification,
        confirmAction,
        formatNumber,
        handleAjaxError
    };
});