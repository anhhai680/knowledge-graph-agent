FROM nginx:alpine

# Copy the HTML file to nginx directory
COPY index.html /usr/share/nginx/html/

# Copy nginx configuration if needed
# COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000

# Configure nginx to run on port 3000
RUN sed -i 's/listen       80;/listen       3000;/' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]
