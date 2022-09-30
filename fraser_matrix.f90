subroutine mat(k, w, R, N, gamma, theta, c_1, c_2, mode, A)
   implicit none
   ! INPUT VARIABLES
   character(len=2), intent(in):: mode
   integer(kind=4), intent(in):: N
   real(kind=8), intent(in) :: k
   real(kind=8), intent(in) :: w
   real(kind=8), intent(in) :: c_1, c_2
   real(kind=8), dimension(:), intent(in) :: R, gamma, theta 
   ! OUTPUT VARIABLES
   complex(kind=8), intent(out), dimension(3*N, 3*N) :: A
   ! INTERNAL VARIABLES
   ! Scalars
   real(kind=8) :: sign
   integer(kind=4) :: i, j, nn, enp2, enp1, enp0, enm2, enm1, nm0, jj, s0, s1, s2
   complex(kind=8) :: KK, alpha, beta, cc2, cc1, c
   ! 1D arrays
   real(kind=8), dimension(N) :: cos_p2, cos_p1, cos_, cos_m1, cos_m2, sin_p2, sin_m2
   complex(kind=8), dimension(N) :: bR, aR, jcp2, jcm2, jsp2, jsm2, jcp1, jcm1
   integer(kind=4), dimension(N) :: pp
   complex(kind=8), dimension(0:2*N+5) :: cdj0, cby0, cdy0, cbj1, cbj !XXX: dimension
   ! 2D arrays
   complex(kind=8), dimension(N, N) :: ja_np2, ja_np1, ja_np0, ja_nm1, ja_nm2
   complex(kind=8), dimension(N, N) :: jb_np2, jb_np1, jb_np0, jb_nm1, jb_nm2
   complex(kind=8), dimension(N, N) :: A11, A12, A13, A21, A22, A23, A31, A32, A33
 
   sign = 0.
   c = w/k
   cc2 = (c/c_2)**2
   cc1 = (c/c_1)**2
   alpha = k*sqrt(cmplx(cc1-1))
   beta =  k*sqrt(cmplx(cc2-1))
   aR = alpha*R
   bR = beta*R
   KK = (1./2.)*(beta**2-k**2)/alpha**2

   select case (mode)
   case ('L', 'T')
      pp = [ (jj, jj=0,2*N-2,2) ]  ! array of even values
      ! end indices
      enp2 = 2*N
      enp1 = 2*N-1
      enp0 = 2*N-2
      enm1 = 2*N-3
      enm2 = 2*N-4
      ! start indices
      s0 = 2
      s1 = 1
      s2 = 0
   case ('Bx', 'By') 
      pp = [ (jj, jj=1,2*N-1,2) ]  ! array of odd values
      ! end indices
      enp2 = 2*N+1
      enp1 = 2*N
      enp0 = 2*N-1
      enm1 = 2*N-2
      enm2 = 2*N-3
      ! start indices
      s0 = 3
      s1 = 2
      s2 = 1
   case default
      print*, "Invalid mode: ", mode
   end select

   ! COMPUTE ALL THE NECESSARY BESSEL FUNCTIONS
   bessel: do i = 1, N
      call cjyna(enp2, bR(i), nm0, cbj, cdj0, cby0, cdy0)
      jb_np2(i,:) = cbj(s0:enp2:2)   ! J_{n+2}(bR)
      jb_np1(i,:) = cbj(s1:enp1:2) ! J_{n+1}(bR)
      jb_np0(i,:) = cbj(s2:enp0:2) ! J_{n}(bR)
      jb_nm1(i,1) = ((-1)**1)* cbj(1) ! J_{-1}
      jb_nm1(i,2:N) = cbj(s1:enm1:2)  ! J_{n-1}(bR)
      jb_nm2(i,1)  = ((-1)**2)* cbj(2) ! J_{-2}
      jb_nm2(i,2:N) = cbj(s2:enm2:2)  ! J_{n-2}(bR)
      
      call cjyna(enp2, aR(i), nm0, cbj1, cdj0, cby0, cdy0)
      ja_np2(i,:)  = cbj1(s0:enp2:2)    ! J_{n+2}(aR)
      ja_np1(i,:) = cbj1(s1:enp1:2)  ! J_{n+1}(aR)
      ja_np0(i,:) = cbj1(s2:enp0:2)  ! J_{n}(aR)
      ja_nm2(i, 2:N) = cbj1(s2:enm2:2)  ! J_{n-2}(aR) 
      ja_nm2(i,1) =  ((-1)**2)* cbj1(2) ! See DLMF eq. 10.4.1
      ja_nm1(i,1) = ((-1)**1)* cbj1(1) ! See DLMF eq. 10.4.1
      ja_nm1(i,2:N) = cbj1(1:enm1:2) ! J_{n-1}(aR)
      
      select case (mode)
      case ('Bx', 'By')
         jb_nm2(i,1) = ((-1)**1)* cbj(1) ! See DLMF eq. 10.4.1
         jb_nm1(i,:) = cbj(0:enm1:2)  ! J_{n-1}(bR)
         ja_nm2(i,1) =  ((-1)**1)* cbj1(1) ! See DLMF eq. 10.4.1
         ja_nm1(i,:) = cbj1(0:enm1:2) ! J_{n-1}(aR)
      case default
         !print*, "Invalid mode: ", mode
      end select
   end do bessel

   ! COMPUTE Aii BLOCKS
   aiiblocks: do j=1,N
      nn = pp(j)
      select case (mode)
      case ('L', 'By')
         ! Terms involving An, Bn, Cn coefficients
         cos_p2 = cos(nn*theta + 2*gamma) 
         cos_m2 = cos(nn*theta - 2*gamma)
         sin_p2 = sin(nn*theta + 2*gamma)
         sin_m2 = sin(nn*theta - 2*gamma)
         cos_p1 = cos(nn*theta + gamma)
         cos_m1 = cos(nn*theta - gamma)
         cos_   = cos(nn*theta)
         sign = 1.
      case ('T', 'Bx') 
         ! Terms involving Dn, En, Fn coefficients
         ! 'sin' becomes 'cos' and vice versa
         cos_p2 = sin(nn*theta + 2*gamma)
         cos_m2 = sin(nn*theta - 2*gamma)
         sin_p2 = cos(nn*theta + 2*gamma)
         sin_m2 = cos(nn*theta - 2*gamma)
         cos_p1 = sin(nn*theta + gamma)
         cos_m1 = sin(nn*theta - gamma)
         cos_   = sin(nn*theta) 
         sign = -1.
      case default
         print*, "Invalid mode : ", mode
      end select

      jcp2 = jb_np2(:,j)*cos_p2
      jcp1 = jb_np1(:,j)*cos_p1
      jcm1 = jb_nm1(:,j)*cos_m1
      jcm2 = jb_nm2(:,j)*cos_m2
      jsp2 = jb_np2(:,j)*sin_p2
      jsm2 = jb_nm2(:,j)*sin_m2

      A11(:, j) = -jcp2 + jcm2
      A12(:, j) =  jcp2 + jcm2 - 2*jb_np0(:,j)*cos_
      A13(:, j) =  ja_np2(:,j)*cos_p2 + ja_nm2(:,j)*cos_m2 - 2*(2*KK-1)*ja_np0(:,j)*cos_
      A21(:, j) = -jsp2 - jsm2
      A22(:, j) =  jsp2 - jsm2
      A23(:, j) =  ja_np2(:,j)*sin_p2 - ja_nm2(:,j)*sin_m2
      A31(:, j) = ((jcp1 + jcm1)*k/beta)*cmplx(0,1)
      A32(:, j) = ((jcp1 - jcm1)*(beta**2-k**2)/(beta*k))*cmplx(0, 1)
      A33(:, j) = ((-ja_np1(:,j)*cos_p1 + ja_nm1(:,j)*cos_m1)*2*k/alpha)*cmplx(0, 1)  
   end do aiiblocks

   ! BUILD A MATRIX
   ! \tau_t
   A(1:N, 1:N)       = A21*sign  ! First raw is zero (ABC, DEF)
   A(1:N, N+1:2*4)   = A22*sign
   A(1:N, 2*N+1:3*N) = A23*sign
   !  \sigma_n
   A(N+1:2*N, 1:N)       = A11
   A(N+1:2*N, N+1:2*N)   = A12
   A(N+1:2*N, 2*N+1:3*N) = A13
   ! \tau_z
   A(2*N+1:3*N, 1:N)       = A31  ! First raw is zero (DEF)
   A(2*N+1:3*N, N+1:2*N)   = A32
   A(2*N+1:3*N, 2*N+1:3*N) = A33
 end subroutine mat
